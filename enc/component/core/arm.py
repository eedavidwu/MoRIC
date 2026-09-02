

from typing import OrderedDict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, index_select, nn


class ArmLinear(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool = False,
    ):

        super().__init__()

        self.residual = residual

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels), requires_grad=True
        )
        self.bias = nn.Parameter(torch.empty((out_channels)), requires_grad=True)
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        self.bias = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)
        if self.residual:
            self.weight = nn.Parameter(
                torch.zeros_like(self.weight), requires_grad=True
            )
        else:
            out_channel = self.weight.size()[0]
            self.weight = nn.Parameter(
                torch.randn_like(self.weight) / out_channel**2, requires_grad=True
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.residual:
            return F.linear(x, self.weight, bias=self.bias) + x

        else:
            return F.linear(x, self.weight, bias=self.bias)

class ArmIntLinear(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fpfm: int = 0,
        pure_int: bool = False,
        residual: bool = False,
    ):

        super().__init__()

        self.fpfm = fpfm
        self.pure_int = pure_int
        self.residual = residual

        if self.pure_int:
            self.weight = nn.Parameter(
                torch.empty((out_channels, in_channels), dtype=torch.int32), requires_grad=False
            )
            self.bias = nn.Parameter(torch.empty((out_channels), dtype=torch.int32), requires_grad=False)
        else:
            self.weight = nn.Parameter(
                torch.empty((out_channels, in_channels), dtype=torch.float), requires_grad=False
            )
            self.bias = nn.Parameter(torch.empty((out_channels), dtype=torch.float), requires_grad=False)


    def forward(self, x: Tensor) -> Tensor:
        if self.residual:
            xx = F.linear(x, self.weight, bias=self.bias) + x*self.fpfm
        else:
            xx = F.linear(x, self.weight, bias=self.bias)

        if self.pure_int:
            xx = xx + torch.sign(xx)*self.fpfm//2
            neg_result = -((-xx)//self.fpfm)
            pos_result = xx//self.fpfm
            result = torch.where(xx < 0, neg_result, pos_result)
        else:
            xx = xx + torch.sign(xx)*self.fpfm/2
            neg_result = -((-xx)/self.fpfm)
            pos_result = xx/self.fpfm
            result = torch.where(xx < 0, neg_result, pos_result)
            result = result.to(torch.int32).to(torch.float)

        return result

class Arm(nn.Module):


    def __init__(self, dim_arm: int, n_hidden_layers_arm: int):
        super().__init__()

        assert dim_arm % 8 == 0, (
            f"ARM context size and hidden layer dimension must be "
            f"a multiple of 8. Found {dim_arm}."
        )

        layers_list = nn.ModuleList()

        for i in range(n_hidden_layers_arm):
            layers_list.append(ArmLinear(dim_arm, dim_arm, residual=True))
            layers_list.append(nn.ReLU())

        layers_list.append(ArmLinear(dim_arm, 2, residual=False))
        self.mlp = nn.Sequential(*layers_list)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        raw_proba_param = self.mlp(x)
        mu = raw_proba_param[:, 0]
        log_scale = raw_proba_param[:, 1]

        scale = torch.exp(torch.clamp(log_scale - 4, min=-4.6, max=5.0))

        return mu, scale, log_scale

    def get_param(self) -> OrderedDict[str, Tensor]:
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
        for layer in self.mlp.children():
            if isinstance(layer, ArmLinear):
                layer.initialize_parameters()

class ArmInt(nn.Module):

    def __init__(self, dim_arm: int, n_hidden_layers_arm: int, fpfm: int, pure_int: bool):
        super().__init__()

        assert dim_arm % 8 == 0, (
            f"ARM context size and hidden layer dimension must be "
            f"a multiple of 8. Found {dim_arm}."
        )

        self.FPFM = fpfm
        self.pure_int = pure_int

        layers_list = nn.ModuleList()

        for i in range(n_hidden_layers_arm):
            layers_list.append(ArmIntLinear(dim_arm, dim_arm, self.FPFM, self.pure_int, residual=True))
            layers_list.append(nn.ReLU())

        layers_list.append(ArmIntLinear(dim_arm, 2, self.FPFM, self.pure_int, residual=False))
        self.mlp = nn.Sequential(*layers_list)

    def set_param_from_float(self, float_param: OrderedDict[str, Tensor]) -> None:

        integerised_param = {}
        for k in float_param:
            if "weight" in k:
                float_v = float_param[k]*self.FPFM
            else:
                float_v = float_param[k]*self.FPFM*self.FPFM

            float_v = float_v + torch.sign(float_v)*0.5
            neg_result = -(-float_v).to(torch.int32)
            pos_result = float_v.to(torch.int32)
            int_v = torch.where(float_v < 0, neg_result, pos_result)
            if not self.pure_int:
                int_v = int_v.to(torch.float)
            integerised_param[k] = nn.parameter.Parameter(int_v, requires_grad=False)

        self.load_state_dict(integerised_param, assign=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        xint = x.clone().detach()
        xint = xint*self.FPFM
        if self.pure_int:
            xint = xint.to(torch.int32)

        for idx_l, layer in enumerate(self.mlp.children()):
            xint = layer(xint)

        raw_proba_param = xint / self.FPFM

        mu = raw_proba_param[:, 0]
        log_scale = raw_proba_param[:, 1]

        scale = torch.exp(torch.clamp(log_scale - 4, min=-4.6, max=5.0))

        return mu, scale, log_scale

    def get_param(self) -> OrderedDict[str, Tensor]:
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        self.load_state_dict(param)

def _get_neighbor(x: Tensor, mask_size: int, non_zero_pixel_ctx_idx: Tensor) -> Tensor:
    pad = int((mask_size - 1) / 2)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)

    x_unfold = (
        x_pad.unfold(2, mask_size, step=1)
        .unfold(3, mask_size, step=1)
        .reshape(-1, mask_size * mask_size)
    )


    neighbor = index_select(x_unfold, dim=1, index=non_zero_pixel_ctx_idx)
    return neighbor


@torch.jit.script
def _laplace_cdf(x: Tensor, expectation: Tensor, scale: Tensor) -> Tensor:
    shifted_x = x - expectation
    return 0.5 - 0.5 * (shifted_x).sign() * torch.expm1(-(shifted_x).abs() / scale)


def _get_non_zero_pixel_ctx_index(dim_arm: int) -> Tensor:

    if dim_arm == 8:
        return torch.tensor(
            [            13,
                         22,
                     30, 31, 32,
             37, 38, 39,
            ]
        )

    elif dim_arm == 16:
        return torch.tensor(
            [
                            13, 14,
                    20, 21, 22, 23, 24,
                28, 29, 30, 31, 32, 33,
                37, 38, 39,
            ]
        )

    elif dim_arm == 24:
        return torch.tensor(
            [
                                4 ,
                        11, 12, 13, 14, 15,
                    19, 20, 21, 22, 23, 24, 25,
                    28, 29, 30, 31, 32, 33, 34,
                36, 37, 38, 39,
            ]
        )

    elif dim_arm == 32:
        return torch.tensor(
            [
                        2 , 3 , 4 , 5 ,
                    10, 11, 12, 13, 14, 15, 16,
                    19, 20, 21, 22, 23, 24, 25, 26,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39,
            ]
        )
