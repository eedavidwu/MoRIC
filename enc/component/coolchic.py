
import math
import typing
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, TypedDict

from torch import nn, Tensor

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from enc.component.core.arm import (
    Arm,
    _get_neighbor,
    _get_non_zero_pixel_ctx_index,
    _laplace_cdf,
)
from enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
    quantize,
)
from enc.component.core.synthesis import Synthesis
from enc.component.core.upsampling import Upsampling
from enc.utils.misc import (
    MAX_ARM_MASK_SIZE,
    POSSIBLE_DEVICE,
    DescriptorCoolChic,
    DescriptorNN,
    measure_expgolomb_rate,
)


@dataclass
class CoolChicEncoderParameter:
    layers_synthesis: List[str]
    n_ft_per_res: List[int]
    dim_arm: int = 24
    n_hidden_layers_arm: int = 2
    upsampling_kernel_size: int = 8
    static_upsampling_kernel: bool = False
    encoder_gain: int = 16

    latent_n_grids: int = field(init=False)
    img_size: Optional[Tuple[int, int]] = field(init=False, default=None)

    def __post_init__(self):
        self.latent_n_grids = len(self.n_ft_per_res)

    def set_image_size(self, img_size: Tuple[int, int]) -> None:
        self.img_size = img_size

    def pretty_string(self) -> str:
        ATTRIBUTE_WIDTH = 25
        VALUE_WIDTH = 80

        s = "CoolChicEncoderParameter value:\n"
        s += "-------------------------------\n"
        for k in fields(self):
            s += f"{k.name:<{ATTRIBUTE_WIDTH}}: {str(getattr(self, k.name)):<{VALUE_WIDTH}}\n"
        s += "\n"
        return s


class CoolChicEncoderOutput(TypedDict):

    raw_out: Tensor
    rate: Tensor
    additional_data: Dict[str, Any]


class CoolChicEncoder(nn.Module):

    def __init__(self, param: CoolChicEncoderParameter):
        super().__init__()

        self.param = param

        assert self.param.img_size is not None, (
            "."
        )

        self.encoder_gains = param.encoder_gain

        self.size_per_latent = []
        self.latent_grids = nn.ParameterList()
        for i in range(self.param.latent_n_grids):
            h_grid, w_grid = [int(math.ceil(x / (2**i))) for x in self.param.img_size]
            c_grid = self.param.n_ft_per_res[i]
            cur_size = (1, c_grid, h_grid, w_grid)

            self.size_per_latent.append(cur_size)

            self.latent_grids.append(
                nn.Parameter(torch.empty(cur_size), requires_grad=True)
            )

        self.initialize_latent_grids()

        self.synthesis = Synthesis(
            sum([latent_size[1] for latent_size in self.size_per_latent]),
            self.param.layers_synthesis,
        )

        self.upsampling = Upsampling(
            self.param.upsampling_kernel_size, self.param.static_upsampling_kernel
        )



        max_mask_size = MAX_ARM_MASK_SIZE
        max_context_pixel = int((max_mask_size**2 - 1) / 2)
        assert self.param.dim_arm <= max_context_pixel, (
            f"You can not have more context pixels "
            f" than {max_context_pixel}. Found {self.param.dim_arm}"
        )

        self.mask_size = max_mask_size

        self.register_buffer(
            "non_zero_pixel_ctx_index",
            _get_non_zero_pixel_ctx_index(self.param.dim_arm),
            persistent=False,
        )

        self.arm = Arm(self.param.dim_arm, self.param.n_hidden_layers_arm)

        self.flops_str = ""
        self.total_flops = 0.0
        self.get_flops()

        self.modules_to_send = [tmp.name for tmp in fields(DescriptorCoolChic)]

        self.nn_q_step: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }

        self.nn_expgol_cnt: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }

        self.full_precision_param = None

    def forward(
        self,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: Optional[float] = 0.3,
        noise_parameter: Optional[float] = 1.0,
        AC_MAX_VAL: int = -1,
        flag_additional_outputs: bool = False,
    ) -> CoolChicEncoderOutput:


        encoder_side_flat_latent = torch.cat(
            [latent_i.view(-1) for latent_i in self.latent_grids]
        )

        flat_decoder_side_latent = quantize(
            encoder_side_flat_latent * self.encoder_gains,
            quantizer_noise_type if self.training else "none",
            quantizer_type if self.training else "hardround",
            soft_round_temperature,
            noise_parameter,
        )

        if AC_MAX_VAL != -1:
            flat_decoder_side_latent = torch.clamp(
                flat_decoder_side_latent, -AC_MAX_VAL, AC_MAX_VAL + 1
            )

        decoder_side_latent = []
        cnt = 0
        for latent_size in self.size_per_latent:
            b, c, h, w = latent_size
            latent_numel = b * c * h * w
            decoder_side_latent.append(
                flat_decoder_side_latent[cnt : cnt + latent_numel].view(latent_size)
            )
            cnt += latent_numel


        flat_context = torch.cat(
            [
                _get_neighbor(spatial_latent_i, self.mask_size, self.non_zero_pixel_ctx_index)
                for spatial_latent_i in decoder_side_latent
            ],
            dim=0,
        )

        flat_latent = torch.cat(
            [spatial_latent_i.view(-1) for spatial_latent_i in decoder_side_latent],
            dim=0
        )

        flat_mu, flat_scale, flat_log_scale = self.arm(flat_context)

        proba = torch.clamp_min(
            _laplace_cdf(flat_latent + 0.5, flat_mu, flat_scale)
            - _laplace_cdf(flat_latent - 0.5, flat_mu, flat_scale),
            min=2**-16,
        )
        flat_rate = -torch.log2(proba)

        synthesis_output = self.synthesis(self.upsampling(decoder_side_latent))

        additional_data = {}
        if flag_additional_outputs:
            additional_data["detailed_sent_latent"] = []
            additional_data["detailed_mu"] = []
            additional_data["detailed_scale"] = []
            additional_data["detailed_log_scale"] = []
            additional_data["detailed_rate_bit"] = []
            additional_data["detailed_centered_latent"] = []

            cnt = 0
            for index_latent_res, _ in enumerate(self.latent_grids):
                c_i, h_i, w_i = decoder_side_latent[index_latent_res].size()[-3:]
                additional_data["detailed_sent_latent"].append(
                    decoder_side_latent[index_latent_res].view((1, c_i, h_i, w_i))
                )

                mu_i, scale_i, log_scale_i, rate_i = [
                    tmp[cnt : cnt + (c_i * h_i * w_i)].view((1, c_i, h_i, w_i))
                    for tmp in [flat_mu, flat_scale, flat_log_scale, flat_rate]
                ]

                cnt += c_i * h_i * w_i
                additional_data["detailed_mu"].append(mu_i)
                additional_data["detailed_scale"].append(scale_i)
                additional_data["detailed_log_scale"].append(log_scale_i)
                additional_data["detailed_rate_bit"].append(rate_i)
                additional_data["detailed_centered_latent"].append(
                    additional_data["detailed_sent_latent"][-1] - mu_i
                )

        res: CoolChicEncoderOutput = {
            "raw_out": synthesis_output,
            "rate": flat_rate,
            "additional_data": additional_data,
        }

        return res

    def get_param(self) -> OrderedDict[str, Tensor]:
        param = OrderedDict({})
        param.update(
            {
                f"latent_grids.{k}": v.detach().clone()
                for k, v in self.latent_grids.named_parameters()
            }
        )
        param.update({f"arm.{k}": v for k, v in self.arm.get_param().items()})
        param.update(
            {f"upsampling.{k}": v for k, v in self.upsampling.get_param().items()}
        )
        param.update(
            {f"synthesis.{k}": v for k, v in self.synthesis.get_param().items()}
        )
        return param

    def set_param(self, param: OrderedDict[str, Tensor]):
        self.load_state_dict(param)

    def initialize_latent_grids(self) -> None:
        for latent_index, latent_value in enumerate(self.latent_grids):
            self.latent_grids[latent_index] = nn.Parameter(
                torch.zeros_like(latent_value), requires_grad=True
            )

    def reinitialize_parameters(self):
        self.arm.reinitialize_parameters()
        self.upsampling.reinitialize_parameters()
        self.synthesis.reinitialize_parameters()
        self.initialize_latent_grids()

        self.nn_q_step: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }
        self.nn_expgol_cnt: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }

    def _store_full_precision_param(self) -> None:

        if self.full_precision_param is not None:
            print(
                "."
            )

        no_q_step = True
        for _, q_step_dict in self.nn_q_step.items():
            for _, q_step in q_step_dict.items():
                if q_step is not None:
                    no_q_step = False
        assert no_q_step, (
            "!"
        )

        no_expgol_cnt = True
        for _, expgol_cnt_dict in self.nn_expgol_cnt.items():
            for _, expgol_cnt in expgol_cnt_dict.items():
                if expgol_cnt is not None:
                    no_expgol_cnt = False
        assert no_expgol_cnt, (
            "!"
        )

        self.full_precision_param = self.get_param()

    def _load_full_precision_param(self) -> None:
        assert self.full_precision_param is not None, (
            "."
        )

        self.set_param(self.full_precision_param)

        self.nn_q_step: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }

        self.nn_expgol_cnt: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }



    def get_flops(self) -> None:
        flops = FlopCountAnalysis(
            self,
            (
                "none",
                "hardround",
                0.3,
                0.1,
                -1,
                False,
            ),
        )
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)

        self.total_flops = flops.total()
        self.flops_str = flop_count_table(flops)
        del flops

    def get_network_rate(self) -> DescriptorCoolChic:
        rate_per_module: DescriptorCoolChic = {
            module_name: {"weight": 0.0, "bias": 0.0}
            for module_name in self.modules_to_send
        }

        for module_name in self.modules_to_send:
            cur_module = getattr(self, module_name)
            rate_per_module[module_name] = measure_expgolomb_rate(
                cur_module,
                self.nn_q_step.get(module_name),
                self.nn_expgol_cnt.get(module_name),
            )

        return rate_per_module

    def get_network_quantization_step(self) -> DescriptorCoolChic:
        return self.nn_q_step

    def get_network_expgol_count(self) -> DescriptorCoolChic:
        return self.nn_expgol_cnt


    def str_complexity(self) -> str:

        if not self.flops_str:
            self.get_flops()

        msg_total_mac = "----------------------------------\n"
        msg_total_mac += (
            f"Total MAC / decoded pixel: {self.get_total_mac_per_pixel():.1f}"
        )
        msg_total_mac += "\n----------------------------------"

        return self.flops_str + "\n\n" + msg_total_mac

    def get_total_mac_per_pixel(self) -> float:

        if not self.flops_str:
            self.get_flops()

        n_pixels = self.param.img_size[-2] * self.param.img_size[-1]
        return self.total_flops / n_pixels

    def to_device(self, device: POSSIBLE_DEVICE) -> None:

        assert device in typing.get_args(
            POSSIBLE_DEVICE
        ), f"Unknown device {device}, should be in {typing.get_args(POSSIBLE_DEVICE)}"
        self = self.to(device)

        for idx_layer, layer in enumerate(self.arm.mlp):
            if hasattr(layer, "qw"):
                if layer.qw is not None:
                    self.arm.mlp[idx_layer].qw = layer.qw.to(device)

            if hasattr(layer, "qb"):
                if layer.qb is not None:
                    self.arm.mlp[idx_layer].qb = layer.qb.to(device)
