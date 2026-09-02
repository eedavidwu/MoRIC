from typing import List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from moric_bistream.armint import MU_LOG_SCALE_MIN_FIXED_POINT, fixed_point_arm
from moric_bistream.constants import (
    FIXED_POINT_DTYPE,
    N_FRAC_BIT_MU_SCALE,
    WEIGHT_SHIFT,
)
from moric_bistream.rangecoder import RangeCoder
from utils.arm import _get_mask_size_ctx, _get_non_zero_pixel_ctx_index


def entropy_coding_latent_arm(
    encoder_data: Optional[Tensor],
    context_inter_features: Optional[Tensor],
    spatial_dim: Tuple[int, int],
    fixed_point_weights: List[Tensor],
    fixed_point_bias: List[Tensor],
    fixed_point_stab_weights: Tensor,
    fixed_point_stab_biases: Tensor,
    range_coder: RangeCoder,
    mode: Literal["encode", "decode"],
    n_spatial_context: int,
) -> Tensor:

    if encoder_data is not None:
        if torch.is_floating_point(encoder_data) or torch.is_complex(encoder_data):
            raise TypeError(
                f"Entropy coded latent should be integer. Found dtype={encoder_data.dtype}"
            )

    ARM_MASK_SIZE = _get_mask_size_ctx()

    h, w = spatial_dim
    coding_order = generate_coding_order((1, h, w), ARM_MASK_SIZE)

    _, occurrence_coding_order = torch.unique(coding_order, return_counts=True)
    max_parallel_decoded = occurrence_coding_order.max()

    offset_index_arm = compute_offset(
        spatial_dim, ARM_MASK_SIZE, _get_non_zero_pixel_ctx_index(n_spatial_context)
    )

    pad = (ARM_MASK_SIZE - 1) // 2
    padded_width = w + 2 * pad

    if encoder_data is not None:
        encoder_data = F.pad(encoder_data, (pad, pad, pad, pad), mode="constant", value=0.0)
        encoder_data = encoder_data.flatten().contiguous()

    if context_inter_features is not None:
        context_inter_features = F.pad(
            context_inter_features, (pad, pad, pad, pad), mode="constant", value=0.0
        )
        context_inter_features = rearrange(context_inter_features, "1 c h w -> (h w) c")

    data_to_fill = torch.zeros((1, 1, h + 2 * pad, w + 2 * pad), dtype=FIXED_POINT_DTYPE)
    data_to_fill = data_to_fill.flatten().contiguous()

    coding_order = F.pad(coding_order, (pad, pad, pad, pad), mode="constant", value=-1)
    coding_order = coding_order.flatten().contiguous()

    all_index_coding = torch.arange(coding_order.max() + 1, dtype=torch.int32)

    if w <= ARM_MASK_SIZE:
        all_idx = [
            (
                (pad + row_idx) * (w + 2 * pad)
                + pad
                + torch.arange(w)
            )
            for row_idx in range(h)
        ]
        all_idx = torch.cat(all_idx).view(-1, 1)
    else:
        all_start_y = torch.zeros_like(all_index_coding, dtype=torch.int32)
        all_start_x = torch.arange(coding_order.max() + 1, dtype=torch.int32)
        all_start_y[w:] = (all_index_coding[w:] - w) // (ARM_MASK_SIZE + 1) + 1
        all_start_x[w:] = w - (ARM_MASK_SIZE + 1) + (all_index_coding[w:] - w) % (ARM_MASK_SIZE + 1)

        i = torch.arange(max_parallel_decoded, dtype=torch.int32).view(1, -1)
        all_start_x_repeat = all_start_x.view(-1, 1).repeat(1, max_parallel_decoded)
        all_start_y_repeat = all_start_y.view(-1, 1).repeat(1, max_parallel_decoded)

        all_idx = padded_width * (pad + all_start_y_repeat + i) + (
            pad + all_start_x_repeat - (ARM_MASK_SIZE + 1) * i
        )

    all_neighbor_idx = (
        all_idx.view(-1, max_parallel_decoded, 1).repeat(1, 1, n_spatial_context) - offset_index_arm
    )

    for index_coding in range(coding_order.max() + 1):
        n_decoded_value = occurrence_coding_order[index_coding]
        idx = all_idx[index_coding, :n_decoded_value]

        neighbor_idx = all_neighbor_idx[index_coding, :n_decoded_value, :].flatten()
        context = torch.index_select(data_to_fill, dim=0, index=neighbor_idx).view(
            -1, n_spatial_context
        )

        if context_inter_features is not None:
            context = torch.cat((context, context_inter_features[idx, :]), dim=1)

        mu_scale = fixed_point_arm(
            context,
            fixed_point_weights,
            fixed_point_bias,
            fixed_point_stab_weights,
            fixed_point_stab_biases,
            output_shift=2 * WEIGHT_SHIFT - N_FRAC_BIT_MU_SCALE,
        )
        idx_mu_scale = mu_scale - MU_LOG_SCALE_MIN_FIXED_POINT

        if mode == "encode":
            range_coder.encode(encoder_data[idx], idx_mu_scale)
            data_to_fill[idx] = encoder_data[idx]

        elif mode == "decode":
            data_to_fill[idx] = range_coder.decode(idx_mu_scale).to(FIXED_POINT_DTYPE)

    if mode == "encode":
        if encoder_data.not_equal(data_to_fill).any():
            raise ValueError(
                "Error when encoding latent. Actually encoded values is not equal "
                "to the expected encoded values. "
                f"a = {encoder_data.abs().sum()}; b = {data_to_fill.abs().sum()}"
            )

    data_to_fill = data_to_fill.reshape(1, 1, h + 2 * pad, w + 2 * pad)
    data_to_fill = data_to_fill[:, :, pad:-pad, pad:-pad]

    return data_to_fill


def compute_offset(
    spatial_dim: Tuple[int, int], mask_size: int, non_zero_pixel_ctx_index: Tensor
) -> Tensor:
    pad = int((mask_size - 1) / 2)

    H, W = spatial_dim
    W_pad = W + 2 * pad

    idx_row = pad - non_zero_pixel_ctx_index // mask_size
    idx_col = pad - non_zero_pixel_ctx_index % mask_size

    offset = idx_col + idx_row * W_pad
    return offset


def generate_coding_order(CHW: Tuple[int, int, int], arm_mask_size: int) -> Tensor:
    C, H, W = CHW

    if W <= arm_mask_size:
        coding_order = torch.arange(0, H * W).view(1, H, W).repeat(C, 1, 1).view(1, C, H, W)
        return coding_order


    first_line = torch.arange(W).view(1, -1).repeat((H, 1))
    row_increment = torch.arange(H) * (arm_mask_size + 1)
    row_increment = row_increment.view(-1, 1)

    coding_order = first_line + row_increment
    coding_order = coding_order.view(1, H, W).repeat(C, 1, 1).view(1, C, H, W)
    return coding_order
