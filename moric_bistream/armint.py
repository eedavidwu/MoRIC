


import copy
import math
from dataclasses import fields
from typing import List, Tuple

import torch
from torch import Tensor

from moric_bistream.constants import (
    FIXED_POINT_DTYPE,
    LOG_SCALE_MIN_FIXED_POINT,
    MU_MIN_FIXED_POINT,
    WEIGHT_SHIFT,
)
from moric_bistream.types import DescriptorNN
from utils.arm import Arm, ArmLinear


def arm_to_fixed_point_param(
    arm: Arm,
    q_steps: DescriptorNN,
    subtract_last_layer: bool = True,
) -> Tuple[List[Tensor], List[Tensor], Tensor, Tensor]:
    fixed_point_weights = []
    fixed_point_biases = []

    fixed_point_param = DescriptorNN(weight=[], bias=[])

    idx_linear_layer = 0
    for i, lay in enumerate(arm.mlp.children()):
        if not isinstance(lay, ArmLinear):
            continue

        is_last_layer = i == (len(list(arm.mlp.children())) - 1)

        for weight_or_bias in fields(DescriptorNN):
            param = getattr(lay, weight_or_bias.name)
            if torch.is_floating_point(param) or torch.is_complex(param):
                raise TypeError(f"Quantized parameter should be integer. Found dtype={param.dtype}")

            if weight_or_bias.name == "bias":
                target_shift = WEIGHT_SHIFT * 2
            elif weight_or_bias.name == "weight":
                target_shift = WEIGHT_SHIFT

            quantize_shift = int(math.log2(q_steps.get_value(weight_or_bias.name)))
            actual_shift = target_shift + quantize_shift

            if is_last_layer and weight_or_bias.name == "bias" and subtract_last_layer:
                param[1] += -(4 << (-quantize_shift))

            fixed_point_param = torch.round(param * (2**actual_shift)).to(FIXED_POINT_DTYPE)

            if weight_or_bias.name == "weight":
                if fixed_point_param.size()[0] == fixed_point_param.size()[1]:
                    identity_matrix = torch.eye(fixed_point_param.size()[0])
                    shift = torch.ones_like(identity_matrix) * target_shift
                    fixed_point_param += (identity_matrix * (2**shift)).to(FIXED_POINT_DTYPE)

                fixed_point_weights.append(fixed_point_param.T)

            elif weight_or_bias.name == "bias":
                fixed_point_biases.append(fixed_point_param)

        idx_linear_layer += 1

    if arm.flag_linear_stabiliser:
        target_shift = WEIGHT_SHIFT
        quantize_shift = int(math.log2(q_steps.get_value("weight")))
        actual_shift = target_shift + quantize_shift

        fixed_point_param = torch.round(arm.stabiliser_branch.weight * (2**actual_shift)).to(
            FIXED_POINT_DTYPE
        )
        fixed_point_weight_stabiliser = fixed_point_param.T

        target_shift = 2 * WEIGHT_SHIFT
        quantize_shift = int(math.log2(q_steps.get_value("bias")))
        actual_shift = target_shift + quantize_shift
        fixed_point_param = torch.round(arm.stabiliser_branch.bias * (2**actual_shift)).to(
            FIXED_POINT_DTYPE
        )
        fixed_point_bias_stabiliser = fixed_point_param

    else:
        assert fixed_point_weights[0].shape[0] == arm.dim_arm, (
            f"arm_to_fixed_point_param's no-stabiliser branch assumes "
            f"context_num == dim_arm (matching bitstream_other's Arm), but "
            f"got first-layer input width {fixed_point_weights[0].shape[0]} "
            f"!= arm.dim_arm {arm.dim_arm}."
        )
        fixed_point_weight_stabiliser = torch.zeros(
            (arm.dim_arm, arm.n_out_features), dtype=FIXED_POINT_DTYPE
        )
        fixed_point_bias_stabiliser = torch.zeros((arm.n_out_features), dtype=FIXED_POINT_DTYPE)

    return (
        fixed_point_weights,
        fixed_point_biases,
        fixed_point_weight_stabiliser,
        fixed_point_bias_stabiliser,
    )


MU_LOG_SCALE_MIN_FIXED_POINT = torch.tensor(
    [MU_MIN_FIXED_POINT, LOG_SCALE_MIN_FIXED_POINT], dtype=FIXED_POINT_DTYPE
)


def fixed_point_arm(
    x: Tensor,
    fixed_point_weights: List[Tensor],
    fixed_point_biases: List[Tensor],
    fixed_point_weights_stab: Tensor,
    fixed_point_biases_stab: Tensor,
    output_shift: int = 0,
) -> Tensor:

    x = x << WEIGHT_SHIFT
    stabiliser = torch.addmm(fixed_point_biases_stab, x, fixed_point_weights_stab)

    for w, b in zip(fixed_point_weights[:-1], fixed_point_biases[:-1]):
        x = (torch.addmm(b, x, w)).clamp_min_(0) >> WEIGHT_SHIFT

    x = torch.addmm(fixed_point_biases[-1], x, fixed_point_weights[-1]) + stabiliser
    return x >> output_shift


def build_int_arm(arm_float: Arm, q_step_weight: float, q_step_bias: float) -> Arm:
    arm_int = copy.deepcopy(arm_float)
    for lay in arm_int.mlp:
        if not isinstance(lay, ArmLinear):
            continue
        with torch.no_grad():
            lay.weight = torch.nn.Parameter(
                torch.round(lay.weight.detach() / q_step_weight).to(FIXED_POINT_DTYPE),
                requires_grad=False,
            )
            lay.bias = torch.nn.Parameter(
                torch.round(lay.bias.detach() / q_step_bias).to(FIXED_POINT_DTYPE),
                requires_grad=False,
            )
    if arm_int.flag_linear_stabiliser:
        with torch.no_grad():
            arm_int.stabiliser_branch.weight = torch.nn.Parameter(
                torch.round(arm_int.stabiliser_branch.weight.detach() / q_step_weight)
                .to(FIXED_POINT_DTYPE),
                requires_grad=False,
            )
            arm_int.stabiliser_branch.bias = torch.nn.Parameter(
                torch.round(arm_int.stabiliser_branch.bias.detach() / q_step_bias)
                .to(FIXED_POINT_DTYPE),
                requires_grad=False,
            )
    return arm_int
