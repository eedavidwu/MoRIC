


import typing
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, OrderedDict, Union

import torch
from enc.component.coolchic import (
    CoolChicEncoder,
    CoolChicEncoderParameter,
)
from enc.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
)
from enc.component.intercoding import InterCodingModule
from torch import Tensor, nn
from enc.utils.codingstructure import (
    FRAME_DATA_TYPE,
    FRAME_TYPE,
    POSSIBLE_BITDEPTH,
    DictTensorYUV,
    convert_444_to_420,
)
from enc.utils.misc import POSSIBLE_DEVICE
from enc.utils.yuv import yuv_dict_clamp


@dataclass
class FrameEncoderOutput:

    decoded_image: Union[Tensor, DictTensorYUV]
    rate: Tensor

    additional_data: Dict[str, Any] = field(default_factory=lambda: {})


class FrameEncoder(nn.Module):

    def __init__(
        self,
        coolchic_encoder_param: CoolChicEncoderParameter,
        frame_type: FRAME_TYPE = "I",
        frame_data_type: FRAME_DATA_TYPE = "rgb",
        bitdepth: POSSIBLE_BITDEPTH = 8,
    ):
        super().__init__()

        self.coolchic_encoder_param = coolchic_encoder_param
        self.frame_type = frame_type
        self.frame_data_type = frame_data_type
        self.bitdepth = bitdepth

        self.coolchic_encoder = CoolChicEncoder(self.coolchic_encoder_param)
        self.inter_coding_module = InterCodingModule(self.frame_type)

    def forward(
        self,
        reference_frames: Optional[List[Tensor]] = None,
        quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
        quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
        soft_round_temperature: Optional[float] = 0.3,
        noise_parameter: Optional[float] = 1.0,
        AC_MAX_VAL: int = -1,
        flag_additional_outputs: bool = False,
    ) -> FrameEncoderOutput:
        coolchic_encoder_output = self.coolchic_encoder.forward(
            quantizer_noise_type=quantizer_noise_type,
            quantizer_type=quantizer_type,
            soft_round_temperature=soft_round_temperature,
            noise_parameter=noise_parameter,
            AC_MAX_VAL=AC_MAX_VAL,
            flag_additional_outputs=flag_additional_outputs,
        )

        inter_coding_output = self.inter_coding_module.forward(
            coolchic_output=coolchic_encoder_output,
            references=[] if reference_frames is None else reference_frames,
            flag_additional_outputs=flag_additional_outputs,
        )

        if self.training:
            decoded_image = inter_coding_output.decoded_image
        else:
            max_dynamic = 2 ** (self.bitdepth) - 1
            decoded_image = (
                torch.round(inter_coding_output.decoded_image * max_dynamic)
                / max_dynamic
            )

        if self.frame_data_type == "yuv420":
            decoded_image = convert_444_to_420(decoded_image)
            decoded_image = yuv_dict_clamp(decoded_image, min_val=0.0, max_val=1.0)
        else:
            decoded_image = torch.clamp(decoded_image, 0.0, 1.0)

        additional_data = {}
        if flag_additional_outputs:
            additional_data.update(coolchic_encoder_output.get("additional_data"))
            additional_data.update(inter_coding_output.additional_data)

        results = FrameEncoderOutput(
            decoded_image=decoded_image,
            rate=coolchic_encoder_output.get("rate"),
            additional_data=additional_data,
        )

        return results

    def get_param(self) -> OrderedDict[str, Tensor]:
        param = OrderedDict({})
        param.update(
            {
                f"coolchic_encoder.{k}": v
                for k, v in self.coolchic_encoder.get_param().items()
            }
        )

        return param

    def set_param(self, param: OrderedDict[str, Tensor]):
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
        self.coolchic_encoder.reinitialize_parameters()

    def set_to_train(self) -> None:
        self = self.train()
        self.coolchic_encoder = self.coolchic_encoder.train()
        self.inter_coding_module = self.inter_coding_module.train()

    def set_to_eval(self) -> None:
        self = self.eval()
        self.coolchic_encoder = self.coolchic_encoder.eval()
        self.inter_coding_module = self.inter_coding_module.eval()

    def to_device(self, device: POSSIBLE_DEVICE) -> None:
        assert device in typing.get_args(
            POSSIBLE_DEVICE
        ), f"Unknown device {device}, should be in {typing.get_args(POSSIBLE_DEVICE)}"

        self = self.to(device)
        self.coolchic_encoder.to_device(device)

    def save(self) -> BytesIO:
        buffer = BytesIO()
        data_to_save = {
            "bitdepth": self.bitdepth,
            "frame_type": self.frame_type,
            "frame_data_type": self.frame_data_type,
            "coolchic_encoder_param": self.coolchic_encoder_param,
            "coolchic_encoder": self.coolchic_encoder.get_param(),
            "coolchic_nn_q_step": self.coolchic_encoder.get_network_quantization_step(),
            "coolchic_nn_expgol_cnt": self.coolchic_encoder.get_network_expgol_count(),
        }

        if self.coolchic_encoder.full_precision_param is not None:
            data_to_save["coolchic_full_precision_param"] = self.coolchic_encoder.full_precision_param

        torch.save(data_to_save, buffer)


        return buffer

def load_frame_encoder(raw_bytes: BytesIO) -> FrameEncoder:
    raw_bytes.seek(0)
    loaded_data = torch.load(raw_bytes, map_location="cpu")

    frame_encoder = FrameEncoder(
        coolchic_encoder_param=loaded_data["coolchic_encoder_param"],
        frame_type=loaded_data["frame_type"],
        frame_data_type=loaded_data["frame_data_type"],
        bitdepth=loaded_data["bitdepth"],
    )

    frame_encoder.coolchic_encoder.set_param(loaded_data["coolchic_encoder"])
    frame_encoder.coolchic_encoder.nn_q_step = loaded_data["coolchic_nn_q_step"]
    if "coolchic_nn_expgol_cnt" in loaded_data:
        frame_encoder.coolchic_encoder.nn_expgol_cnt = loaded_data["coolchic_nn_expgol_cnt"]

    if "coolchic_full_precision_param" in loaded_data:
        frame_encoder.coolchic_encoder.full_precision_param = loaded_data["coolchic_full_precision_param"]

    return frame_encoder
