

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn

from enc.component.coolchic import CoolChicEncoderOutput
from enc.utils.codingstructure import FRAME_TYPE


def warp(x, flo, interpol_mode="bilinear", padding_mode="border", align_corners=True):
    B, C, H, W = x.size()
    cur_device = x.device

    xx = torch.arange(0, W, device=cur_device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=cur_device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    vgrid = torch.autograd.Variable(grid) + flo

    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(
        x,
        vgrid,
        mode=interpol_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    mask = torch.autograd.Variable(torch.ones(x.size(), device=cur_device))
    mask = nn.functional.grid_sample(
        mask,
        vgrid,
        mode=interpol_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


@dataclass
class InterCodingModuleInput:

    residue: Optional[Tensor] = (
        None
    )
    flow_1: Optional[Tensor] = (
        None
    )
    flow_2: Optional[Tensor] = (
        None
    )
    alpha: Optional[Tensor] = (
        None
    )
    beta: Optional[Tensor] = (
        None
    )


@dataclass
class InterCodingModuleOutput:

    decoded_image: Tensor

    additional_data: Dict[str, Any] = field(default_factory=lambda: {})


class InterCodingModule(nn.Module):
    def __init__(self, frame_type: FRAME_TYPE):
        super().__init__()

        self.frame_type = frame_type

        self.flow_gain = 1.0

    def process_coolchic_output(
        self, coolchic_output: CoolChicEncoderOutput
    ) -> InterCodingModuleInput:
        raw_coolchic_output = coolchic_output.get("raw_out")
        residue = raw_coolchic_output[:, :3, :, :]


        if self.frame_type == "P" or self.frame_type == "B":
            flow_1 = raw_coolchic_output[:, 3:5, :, :] * self.flow_gain
            alpha = torch.clamp(raw_coolchic_output[:, 5:6, :, :] + 0.5, 0.0, 1.0)
        else:
            flow_1 = None
            alpha = None

        if self.frame_type == "B":
            flow_2 = raw_coolchic_output[:, 6:8, :, :] * self.flow_gain
            beta = torch.clamp(raw_coolchic_output[:, 8:9, :, :] + 0.5, 0.0, 1.0)
        else:
            flow_2 = None
            beta = None

        return InterCodingModuleInput(
            residue=residue, flow_1=flow_1, flow_2=flow_2, alpha=alpha, beta=beta
        )

    def forward(
        self,
        coolchic_output: CoolChicEncoderOutput,
        references: List[Tensor],
        flag_additional_outputs: bool = False,
    ) -> InterCodingModuleOutput:
        input_inter_coding = self.process_coolchic_output(coolchic_output)

        if self.frame_type == "I":
            decoded_frame = input_inter_coding.residue
        else:
            if self.frame_type == "P":
                prediction = warp(references[0], input_inter_coding.flow_1)
            if self.frame_type == "B":
                prediction = input_inter_coding.beta * warp(
                    references[0], input_inter_coding.flow_1
                ) + (1 - input_inter_coding.beta) * warp(
                    references[1], input_inter_coding.flow_2
                )

            masked_prediction = input_inter_coding.alpha * prediction
            decoded_frame = masked_prediction + input_inter_coding.residue

        additional_data = {}
        if flag_additional_outputs:
            additional_data["residue"] = input_inter_coding.residue

            if self.frame_type == "P" or self.frame_type == "B":
                additional_data["alpha"] = input_inter_coding.alpha
                additional_data["flow_1"] = input_inter_coding.flow_1
                additional_data["prediction"] = prediction
                additional_data["masked_prediction"] = masked_prediction

            if self.frame_type == "B":
                additional_data["beta"] = input_inter_coding.beta
                additional_data["flow_2"] = input_inter_coding.flow_2

        return InterCodingModuleOutput(
            decoded_image=decoded_frame, additional_data=additional_data
        )
