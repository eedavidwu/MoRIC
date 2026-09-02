

import os

import numpy as np
import torch
from einops import rearrange
from enc.utils.codingstructure import (
    FRAME_DATA_TYPE,
    POSSIBLE_BITDEPTH,
    DictTensorYUV,
    FrameData,
)
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import to_tensor


def yuv_dict_clamp(yuv: DictTensorYUV, min_val: float, max_val: float) -> DictTensorYUV:
    clamped_yuv = DictTensorYUV(
        y=yuv.get("y").clamp(min_val, max_val),
        u=yuv.get("u").clamp(min_val, max_val),
        v=yuv.get("v").clamp(min_val, max_val),
    )
    return clamped_yuv


def load_frame_data_from_file(filename: str, idx_display_order: int) -> FrameData:

    if filename.endswith(".yuv"):
        bitdepth: POSSIBLE_BITDEPTH = 8 if "_8b" in filename else 10
        frame_data_type: FRAME_DATA_TYPE = "yuv420" if "420" in filename else "yuv444"
        data = read_yuv(filename, idx_display_order, frame_data_type, bitdepth)

    elif filename.endswith(".png"):
        bitdepth: POSSIBLE_BITDEPTH = 8
        frame_data_type: FRAME_DATA_TYPE = "rgb"
        data = to_tensor(Image.open(filename))
        data = rearrange(data, "c h w -> 1 c h w")

    return FrameData(bitdepth, frame_data_type, data)


def read_yuv(filename: str, frame_idx: int, frame_data_type: FRAME_DATA_TYPE, bit_depth: POSSIBLE_BITDEPTH) -> DictTensorYUV:

    w, h = [
        int(tmp_str)
        for tmp_str in os.path.basename(filename).split(".")[0].split("_")[1].split("x")
    ]

    if frame_data_type == "yuv420":
        w_uv, h_uv = [int(x / 2) for x in [w, h]]
    else:
        w_uv, h_uv = w, h

    byte_per_value = 1 if bit_depth == 8 else 2

    n_val_y = h * w
    n_val_uv = h_uv * w_uv
    n_val_per_frame = n_val_y + 2 * n_val_uv

    n_bytes_y = n_val_y * byte_per_value
    n_bytes_uv = n_val_uv * byte_per_value
    n_bytes_per_frame = n_bytes_y + 2 * n_bytes_uv

    raw_video = torch.tensor(
        np.memmap(
            filename,
            mode="r",
            shape=n_val_per_frame,
            offset=n_bytes_per_frame * frame_idx,
            dtype=np.uint16 if bit_depth == 10 else np.uint8,
        ).astype(np.float32)
    )

    ptr = 0
    y = raw_video[ptr : ptr + n_val_y].view(1, 1, h, w)
    ptr += n_val_y
    u = raw_video[ptr : ptr + n_val_uv].view(1, 1, h_uv, w_uv)
    ptr += n_val_uv
    v = raw_video[ptr : ptr + n_val_uv].view(1, 1, h_uv, w_uv)

    norm_factor = 2**bit_depth - 1

    if frame_data_type == "yuv420":
        video = DictTensorYUV(y=y / norm_factor, u=u / norm_factor, v=v / norm_factor)
    else:
        video = torch.cat([y, u, v], dim=1) / norm_factor

    return video


def write_yuv(data: FrameData, filename: str, norm: bool = True) -> None:
    assert data.frame_data_type in ["yuv420", "yuv444"], (
        "Found incorrect datatype in "
        f'write_yuv() function: {data.frame_data_type}. Data type should be "yuv420" or "yuv444".'
    )

    if not (filename[-4:] == ".yuv"):
        filename += ".yuv"
    filename = filename[:-4]

    DUMMY_FRAMERATE = 1
    h, w = data.img_size
    filename = f"{filename}_{w}x{h}_{DUMMY_FRAMERATE}fps_{data.frame_data_type}p_{data.bitdepth}b.yuv"

    if data.frame_data_type == "yuv420":
        raw_data = torch.cat([channels.flatten() for _, channels in data.data.items()])
    elif data.frame_data_type == "yuv444":
        raw_data = data.data.flatten()

    if norm:
        raw_data = raw_data * (2**data.bitdepth - 1)

    dtype = np.uint16 if data.bitdepth == 10 else np.uint8

    raw_data = torch.round(raw_data).cpu().numpy().astype(dtype)

    np.memmap.tofile(raw_data, filename)


def rgb2yuv(rgb: Tensor) -> Tensor:
    assert (
        len(rgb.size()) == 4
    ), f"rgb2yuv input must be a 4D tensor [B, 3, H, W]. Data size: {rgb.size()}"
    assert (
        rgb.size()[1] == 3
    ), f"rgb2yuv input must have 3 channels. Data size: {rgb.size()}"

    r, g, b = rgb.split(1, dim=1)

    y = torch.round(0.299 * r + 0.587 * g + 0.114 * b)
    u = torch.round(-0.1687 * r - 0.3313 * g + 0.5 * b + +128)
    v = torch.round(0.5 * r - 0.4187 * g - 0.0813 * b + 128)

    yuv = torch.cat((y, u, v), dim=1)
    return yuv


def yuv2rgb(yuv: Tensor):
    assert (
        len(yuv.size()) == 4
    ), f"yuv2rgb input must be a 4D tensor [B, 3, H, W]. Data size: {yuv.size()}"
    assert (
        yuv.size()[1] == 3
    ), f"yuv2rgb input must have 3 channels. Data size: {yuv.size()}"

    y, u, v = yuv.split(1, dim=1)
    r = (
        1.0 * y
        + -0.000007154783816076815 * u
        + 1.4019975662231445 * v
        - 179.45477266423404
    )
    g = 1.0 * y + -0.3441331386566162 * u + -0.7141380310058594 * v + 135.45870971679688
    b = (
        1.0 * y
        + 1.7720025777816772 * u
        + 0.00001542569043522235 * v
        - 226.8183044444304
    )
    rgb = torch.cat((r, g, b), dim=1)
    return rgb
