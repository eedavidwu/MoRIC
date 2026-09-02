

import os
import struct
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from enc.utils.misc import POSSIBLE_Q_STEP

from moric_bistream.armint import arm_to_fixed_point_param, build_int_arm
from moric_bistream.constants import AC_MAX_VAL
from moric_bistream.expgolomb import decode_exp_golomb, encode_exp_golomb
from moric_bistream.latent import entropy_coding_latent_arm
from moric_bistream.rangecoder import RangeCoder
from moric_bistream.types import DescriptorNN

_MAGIC = b"MRC6"


def _module_q_symbols(module, module_name: str, nn_q_step: Dict) -> Dict[str, Tuple[List[int], int]]:
    
    out = {}
    param = module.get_param()
    for kind in ("weight", "bias"):
        possible = POSSIBLE_Q_STEP[module_name][kind]
        q_step = nn_q_step[module_name][kind]
        q_idx = int(torch.argmin((possible - q_step).abs()).item())
        q = float(possible[q_idx])
        syms = [
            torch.round(v.detach().cpu() / q).flatten()
            for k, v in param.items()
            if (f".{kind}" in k or k.endswith(kind))
        ]
        flat = torch.cat(syms) if syms else torch.zeros(0)
        out[kind] = (flat.to(torch.int64).tolist(), q_idx)
    return out


def _set_module_from_q_symbols(module, module_name: str, kind_syms: Dict[str, Tuple[List[int], int]]):
    
    param = module.get_param()
    cursor = {"weight": 0, "bias": 0}
    new_param = {}
    for k, v in param.items():
        kind = "weight" if (".weight" in k or k.endswith("weight")) else "bias"
        syms, q_idx = kind_syms[kind]
        q = float(POSSIBLE_Q_STEP[module_name][kind][q_idx])
        n = v.numel()
        vals = torch.tensor(syms[cursor[kind]:cursor[kind] + n], dtype=torch.float32) * q
        cursor[kind] += n
        new_param[k] = vals.view(v.shape).to(v.device)
    module.set_param(new_param)


# ======================================================================== #
#                              Encode / decode                             #
# ======================================================================== #
@torch.no_grad()
def encode_frame_moric(model, bitstream_path: str) -> Dict:
    """Write the trained + quantized model to a single bitstream file:
    [header | NN weights (Cool-Chic 5 exp-Golomb) | latents (Cool-Chic 5
    fixed-point-ARM range coding)].

    Returns a dict with per-part byte counts and the encoder-side quantized
    latents (for verification)."""
    torch.use_deterministic_algorithms(True)
    try:
        model.eval()

       
        q_list = []
        n_clamped = 0
        for lat in model.modulation_sf:
            q = torch.round(lat.detach() * model.encoder_gains_sf)
            q_clamped = torch.clamp(q, -AC_MAX_VAL, AC_MAX_VAL - 1)
            n_clamped += int((q_clamped != q).sum())
            q_list.append(q_clamped.cpu().view(*q_clamped.shape[-2:]).to(torch.int64))
        if n_clamped > 0:
            print(f"[bitstream] WARNING: {n_clamped} latent value(s) clamped to "
                  f"[-{AC_MAX_VAL}, {AC_MAX_VAL - 1}] -- some information was lost.")

      
        import copy
        arm_cpu = copy.deepcopy(model.arm).cpu().eval()
        arm_q_steps = DescriptorNN(
            weight=model.nn_q_step["arm"]["weight"], bias=model.nn_q_step["arm"]["bias"]
        )
        arm_int = build_int_arm(arm_cpu, arm_q_steps.weight, arm_q_steps.bias)
        fp_w, fp_b, fp_w_stab, fp_b_stab = arm_to_fixed_point_param(arm_int, arm_q_steps)

      
        range_coder = RangeCoder()
        n_spatial_context = _arm_context_num(model)
        for q in q_list:
            h, w = q.shape
            entropy_coding_latent_arm(
                q.view(1, 1, h, w), None, (h, w),
                fp_w, fp_b, fp_w_stab, fp_b_stab,
                range_coder, mode="encode", n_spatial_context=n_spatial_context,
            )
        latent_blob = range_coder.get_bitstream_bytes()

      
        nn_meta = []  # (module, kind, q_idx, cnt, n_values)
        all_q_param: List[int] = []
        all_count: List[int] = []
        for module_name in model.modules_to_send:
            module = getattr(model, module_name)
            kind_syms = _module_q_symbols(module, module_name, model.nn_q_step)
            for kind in ("weight", "bias"):
                syms, q_idx = kind_syms[kind]
                cnt = int(model.nn_expgol_cnt[module_name][kind])
                all_q_param += syms
                all_count += [cnt for _ in syms]
                nn_meta.append((module_name, kind, q_idx, cnt, len(syms)))
        nn_blob, nn_n_bit_pad = encode_exp_golomb(all_q_param, all_count)

        # ---- header ----
        L = len(q_list)
        header = bytearray()
        header += _MAGIC
        header += struct.pack(
            "<HHBBB", q_list[-1].shape[0], q_list[-1].shape[1],
            L, len(model.modules_to_send), nn_n_bit_pad,
        )
        for _, _, q_idx, cnt, n_values in nn_meta:
            header += struct.pack("<bbI", q_idx, cnt, n_values)
        header += struct.pack("<I", len(nn_blob))
        header += struct.pack("<I", len(latent_blob))

        with open(bitstream_path, "wb") as f:
            f.write(bytes(header))
            f.write(bytes(nn_blob))
            f.write(latent_blob)

        return {
            "path": bitstream_path,
            "header_bytes": len(header),
            "nn_bytes": len(nn_blob),
            "latent_bytes": len(latent_blob),
            "total_bytes": os.path.getsize(bitstream_path),
            "q_list": [q.float() for q in q_list],
        }
    finally:
        torch.use_deterministic_algorithms(False)


def _arm_context_num(model) -> int:
   
    first_layer = next(m for m in model.arm.mlp if m.__class__.__name__ == "ArmLinear")
    return first_layer.weight.shape[1]


@torch.no_grad()
def decode_frame_moric(bitstream_path: str, shell, coords) -> Dict:
   
    torch.use_deterministic_algorithms(True)
    try:
        shell.eval()
        with open(bitstream_path, "rb") as f:
            data = f.read()

        assert data[:4] == _MAGIC, "Bad magic in bitstream"
        off = 4
        H_last, W_last, L, n_modules, nn_n_bit_pad = struct.unpack_from("<HHBBB", data, off)
        off += struct.calcsize("<HHBBB")
        assert n_modules == len(shell.modules_to_send)

        nn_meta = []
        for _ in range(n_modules * 2):
            q_idx, cnt, n_values = struct.unpack_from("<bbI", data, off)
            off += struct.calcsize("<bbI")
            nn_meta.append((q_idx, cnt, n_values))
        (nn_blob_len,) = struct.unpack_from("<I", data, off)
        off += struct.calcsize("<I")
        (latent_blob_len,) = struct.unpack_from("<I", data, off)
        off += struct.calcsize("<I")

       
        nn_bytes = data[off:off + nn_blob_len]
        off += nn_blob_len
        all_count = [cnt for (_, cnt, n_values) in nn_meta for _ in range(n_values)]
        all_param = decode_exp_golomb(nn_bytes, nn_n_bit_pad, all_count)

        cursor = 0
        meta_by_module = {}
        mi = 0
        for module_name in shell.modules_to_send:
            meta_by_module[module_name] = {}
            for kind in ("weight", "bias"):
                q_idx, cnt, n_values = nn_meta[mi]
                mi += 1
                syms = all_param[cursor:cursor + n_values]
                cursor += n_values
                meta_by_module[module_name][kind] = (syms, q_idx)

        for module_name in shell.modules_to_send:
            module = getattr(shell, module_name)
            _set_module_from_q_symbols(module, module_name, meta_by_module[module_name])

        
        import copy
        arm_cpu = copy.deepcopy(shell.arm).cpu().eval()
        arm_q_steps = DescriptorNN(
            weight=shell.nn_q_step["arm"]["weight"], bias=shell.nn_q_step["arm"]["bias"]
        )
        arm_int = build_int_arm(arm_cpu, arm_q_steps.weight, arm_q_steps.bias)
        fp_w, fp_b, fp_w_stab, fp_b_stab = arm_to_fixed_point_param(arm_int, arm_q_steps)

        range_coder = RangeCoder()
        range_coder.load_bitstream(data[off:off + latent_blob_len])
        off += latent_blob_len

        shapes = [tuple(lat.shape[-2:]) for lat in shell.modulation_sf]
        q_list = []
        for s in range(L):
            h, w = shapes[s]
            grid = entropy_coding_latent_arm(
                None, None, (h, w), fp_w, fp_b, fp_w_stab, fp_b_stab,
                range_coder, mode="decode", n_spatial_context=_arm_context_num(shell),
            )
            q_list.append(grid.view(h, w).float())

       
        device = next(shell.conv_mod.parameters()).device
        q_dev = [q.view(1, 1, *q.shape).to(device) for q in q_list]
        dense = shell.upsampling_2d(q_dev)
        recon = shell.conv_mod(coords.to(device), dense, shell.region_mask_sf[0])

        return {"recon": recon, "q_list": q_list}
    finally:
        torch.use_deterministic_algorithms(False)


# ======================================================================== #
#                        Encode + decode + verify                          #
# ======================================================================== #
@torch.no_grad()
def encode_decode_verify(model, coords, bitstream_path: str) -> Dict:
   
    import copy

    enc = encode_frame_moric(model, bitstream_path)

  
    shell = copy.deepcopy(model)
    for module_name in shell.modules_to_send:
        module = getattr(shell, module_name)
        module.set_param({k: torch.zeros_like(v) for k, v in module.get_param().items()})

    dec = decode_frame_moric(bitstream_path, shell, coords)

   
    latents_match = all(
        torch.equal(qe, qd) for qe, qd in zip(enc["q_list"], dec["q_list"])
    )
    nn_match = True
    for module_name in model.modules_to_send:
        p_enc = getattr(model, module_name).get_param()
        p_dec = getattr(shell, module_name).get_param()
        for k in p_enc:
            if not torch.equal(p_enc[k].cpu(), p_dec[k].cpu()):
                nn_match = False

    return {
        **{k: v for k, v in enc.items() if k != "q_list"},
        "recon": dec["recon"],
        "latents_match": latents_match,
        "nn_match": nn_match,
    }
