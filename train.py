import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#K=N
# nohup python -u train.py --write_bitstream --lambda_rate_list 1e-3 --use_candidate --start_index 0 --end_index 24 > kodak_cc5_1e3_kodim01_24_autoregion_write_bitstream.out&
import re
import math
import random
import argparse
import glob
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from PIL import Image

from models.model import Masked_INR
from models.candidate_train import train_with_candidates
from utils.quantizemodel import quantize_model
from utils.rdoq import rdoq_model
from utils.eval_model import compute_model_rate
from lossy_contour_algorithm import encode_one_mask

manual_seed = 1
def seed_everything(seed=1029):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PATHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(1)

print('seed', manual_seed)


def _kodim_idx_from_path(path):
    m = re.search(r'kodim(\d{2})', os.path.basename(path))
    if m:
        return m.group(1)
    m = re.search(r'kodim(\d{2})', path)
    return m.group(1) if m else None


def extract_individual_region_masks(mask_path, min_area=50):
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'Cannot read mask {mask_path}')
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    masks_tensor_list = []
    all_foreground_mask = np.zeros_like(binary, dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        region_mask = np.zeros_like(binary, dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, color=255, thickness=-1)
        all_foreground_mask = cv2.bitwise_or(all_foreground_mask, region_mask)
        masks_tensor_list.append(torch.from_numpy(region_mask > 0).unsqueeze(0).unsqueeze(0))

    background_mask = (all_foreground_mask == 0)
    masks_tensor_list.append(torch.from_numpy(background_mask).unsqueeze(0).unsqueeze(0))

    print(f"[mask] {mask_path}: {len(masks_tensor_list)} regions (incl. background)")
    return masks_tensor_list


def _generate_lossy_mask(lossless_path, lossy_path, T_init=5, thread=50, rate=0.15,
                          min_area=50):
    
    mask = cv2.imread(lossless_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'Cannot read mask {lossless_path}')
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    os.makedirs(os.path.dirname(lossy_path), exist_ok=True)
    tmp_path = lossy_path + '.tmp_region.png'
    merged = np.zeros_like(binary, dtype=np.uint8)
    total_bits = 0
    n_encoded = 0
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        region_mask = np.zeros_like(binary, dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, color=255, thickness=-1)
        cv2.imwrite(tmp_path, region_mask)
        bits, orig_bits, err, recon = encode_one_mask(
            tmp_path, T_init=T_init, thread=thread, rate=rate)
        merged = cv2.bitwise_or(merged, (recon > 0).astype(np.uint8) * 255)
        total_bits += int(bits)
        n_encoded += 1
        print(f'[lossy] {os.path.basename(lossless_path)} region {n_encoded}: '
              f'{bits} bits (orig {orig_bits}, err {err}px)')
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    cv2.imwrite(lossy_path, merged)
    with open(lossy_path.replace('.png', '_bits.txt'), 'w') as f:
        f.write(f'{total_bits}\n')
    print(f'[lossy] saved {lossy_path}  ({n_encoded} regions, {total_bits} contour bits)')
    return total_bits


def _get_lossy_mask(mask_dir, lossy_mask_dir, idx_str, regen=False):
    
    lossless_path = os.path.join(mask_dir, f'kodim{idx_str}.png')
    lossy_path = os.path.join(lossy_mask_dir, f'kodim{idx_str}.png')
    bits_path = lossy_path.replace('.png', '_bits.txt')

    if regen or not os.path.exists(lossy_path) or not os.path.exists(bits_path):
        total_bits = _generate_lossy_mask(lossless_path, lossy_path)
    else:
        with open(bits_path) as f:
            total_bits = int(f.read().strip())
        print(f'[lossy] reusing {lossy_path} ({total_bits} contour bits)')
    return lossy_path, total_bits


def _region_map_from_mask(lossy_mask_path, H, W):
    
    masks_tensor_list = extract_individual_region_masks(lossy_mask_path)
    K = len(masks_tensor_list)
    fg = masks_tensor_list[:-1]
    fg.sort(key=lambda m: -int(m.sum().item()))

    region_map = np.full((H, W), K - 1, dtype=np.int64)
    for label, m in enumerate(fg):
        m_np = m.squeeze(0).squeeze(0).numpy()
        if m_np.shape != (H, W):
            raise ValueError(f'{lossy_mask_path} shape {m_np.shape} != image ({H}, {W})')
        region_map[m_np] = label
    return torch.from_numpy(region_map).long(), K


def get_mgrid(w, h):
    x = torch.linspace(-1, 1, steps=w)
    y = torch.linspace(-1, 1, steps=h)
    grid = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1)
    return grid.unsqueeze(0).permute(0, 3, 2, 1)


def compute_psnr_ssim(target, recon):
    img = np.transpose(target[0].detach().cpu().numpy(), (1, 2, 0))
    rec = np.transpose(recon[0].detach().cpu().numpy(), (1, 2, 0))
    H, W, _ = img.shape
    win = min(7, H, W)
    if win % 2 == 0:
        win -= 1
    return (
        psnr_fn(img, rec, data_range=1.0),
        ssim_fn(img, rec, data_range=1.0, channel_axis=-1, win_size=win),
    )


def loss_to_psnr(mse):
    return -10.0 * np.log10(max(float(mse), 1e-12))


def encoder_gain_from_lmbda(lmbda):
    
    if lmbda < 0.0002:
        return 24
    if lmbda < 0.0005:
        return 20
    return 16


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")
    return 0


def _format_lambda(lam):
    """0.004 -> '4e3', 0.001 -> '1e3', 0.0015 -> '1.5e3'."""
    s = f"{lam:.6e}"  # e.g. '4.000000e-03'
    mantissa, exp = s.split('e')
    mantissa = mantissa.rstrip('0').rstrip('.')
    if mantissa == '' or mantissa == '-':
        mantissa = mantissa + '0'
    return f"{mantissa}e{abs(int(exp))}"


def build_model(region_map, args, device):
    model_args = SimpleNamespace(
        context_arm=args.context_arm,
        dim_arm_mod=args.dim_arm_mod,
        mod_base=args.mod_base,
        sythesis_features=args.sythesis_features,
        local_upsampling_kernel_size=args.local_upsampling_kernel_size,
        upsampling_kernel_size=args.upsampling_kernel_size,
        upsampling_preconcat_kernel_size=args.upsampling_preconcat_kernel_size,
        static_upsampling_kernel=False,
        highest_flag=1,
        latent_factor=1,
        batch_size=1,
        scale=1,
        encoder_gain=encoder_gain_from_lmbda(args.lambda_rate),
    )
    K = int(args.num_regions)
    model = Masked_INR(
        model_args, region_map, sparsity=0,
        in_features=2, out_features=3,
        hidden_features=24, hidden_layers=0,
        num_regions=K,
    ).to(device)
    return model


def _linear_schedule(initial_final_value, cur_itr, max_itr):
    
    initial_value, final_value = initial_final_value
    return cur_itr * (final_value - initial_value) / max_itr + initial_value


@torch.no_grad()
def _hardround_test(model, coords, target_pixels, total_pixels, lmbda, criterion):
    
    rate_per_module = model.get_network_rate()
    total_rate_nn_bit = sum(v['weight'] + v['bias'] for v in rate_per_module.values())

    model.eval()
    recon_flat, rate_pr, _ = model(coords)
    mse = criterion(recon_flat, target_pixels)
    rate_latent_bit = rate_pr.sum()
    rate_bpp = (rate_latent_bit + total_rate_nn_bit) / total_pixels
    loss = mse + lmbda * rate_bpp
    model.train()

    mse_v = mse.item()
    return {
        'loss': loss.item(),
        'mse': mse_v,
        'psnr': -10.0 * math.log10(max(mse_v, 1e-12)),
        'lat_bpp': rate_latent_bit.item() / total_pixels,
        'nn_bpp': total_rate_nn_bit / total_pixels,
    }


def train(model, img_tensor, coords, args, img_index, saved_path, device,
          contour_bits=0, recon_path=None):
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    total_pixels = H * W
    target_pixels = img_tensor.permute(0, 2, 3, 1).reshape(1, -1, 3)
    criterion = nn.MSELoss()
    all_parameters = list(model.parameters())

    print(f'\n=== MoRIC setup (image {img_index}) ===')
    print(f'  Params        : {sum(p.numel() for p in all_parameters)} ({len(all_parameters)} tensors)')
    print(f'  Optimizer     : Adam (all params) + CosineAnnealingLR w/ patience, hardround-validation best tracking')
    print(f'  Lambda        : {args.lambda_rate}')
    print(f'  Stages        : {args.stage1_steps} (softround+gaussian) + 1500 (STE) steps')

    
    quantizer_type = 'softround'
    quantizer_noise_type = 'gaussian'
    T0, T1 = 0.3, 0.1
    n0, n1 = 0.25, 0.1
    max_itr = args.stage1_steps
    patience = 5000

    model.quantizer_type = quantizer_type
    model.quantizer_noise_type = quantizer_noise_type
    encoder_logs_best = _hardround_test(model, coords, target_pixels, total_pixels, args.lambda_rate, criterion)
    best_state = model.get_param()

    optimizer = torch.optim.Adam(all_parameters, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(max_itr / args.freq_valid, 1), eta_min=1e-5)

    print(f'\n===== STAGE 1: {max_itr} steps (softround, gaussian noise {n0}->{n1}, '
          f'T {T0}->{T1}, lr={args.lr}, patience={patience}) =====')
    print(f'{"itr":>7} {"loss e-3":>10} {"psnr_db":>9} {"lat_bpp":>8} {"nn_bpp":>8} '
          f'{"lr":>10} {"T":>7} {"noise":>7}  record')

    cur_t = _linear_schedule((T0, T1), 0, max_itr)
    cur_n = _linear_schedule((n0, n1), 0, max_itr)
    model.train()
    cnt_record = 0
    for cnt in range(max_itr):
        if cnt - cnt_record > patience:
            model.set_param(best_state)
            current_lr = scheduler.get_last_lr()[0]
            for g in optimizer.param_groups:
                g['lr'] = current_lr
            cnt_record = cnt

        for p in all_parameters:
            p.grad = None

        model.soft_round_temperature = cur_t
        model.noise_parameter = cur_n
        recon_flat, rate_pr, _ = model(coords)
        mse = criterion(recon_flat, target_pixels)
        bpp = rate_pr.sum() / total_pixels
        loss = mse + args.lambda_rate * bpp
        loss.backward()
        nn.utils.clip_grad_norm_(all_parameters, 0.1, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        if (cnt + 1) % args.freq_valid == 0 or (cnt + 1) == max_itr:
            encoder_logs = _hardround_test(model, coords, target_pixels, total_pixels, args.lambda_rate, criterion)
            record = ''
            if encoder_logs['loss'] < encoder_logs_best['loss']:
                best_state = model.get_param()
                encoder_logs_best = encoder_logs
                cnt_record = cnt
                record = '  record'
            cur_lr = scheduler.get_last_lr()[0]
            print(f'{cnt + 1:7d} {encoder_logs["loss"] * 1e3:10.4f} {encoder_logs["psnr"]:9.3f} '
                  f'{encoder_logs["lat_bpp"]:8.4f} {encoder_logs["nn_bpp"]:8.4f} '
                  f'{cur_lr:10.2e} {float(cur_t):7.3f} {float(cur_n):7.3f}{record}')

            cur_t = _linear_schedule((T0, T1), cnt, max_itr)
            cur_n = _linear_schedule((n0, n1), cnt, max_itr)
            scheduler.step()
            model.train()

    model.set_param(best_state)
    print(f'Reloaded best STAGE 1 model (loss={encoder_logs_best["loss"]:.6f}, '
          f'psnr={encoder_logs_best["psnr"]:.3f}dB).')

    if recon_path is not None:
        with torch.no_grad():
            model.eval()
            recon_flat, _, _ = model(coords)
            model.train()
        recon_img_s1 = recon_flat.view(1, H, W, 3).permute(0, 3, 1, 2).clamp(0, 1)
        os.makedirs(os.path.dirname(recon_path), exist_ok=True)
        vutils.save_image(recon_img_s1, recon_path)

  
    quantizer_type = 'ste'
    quantizer_noise_type = 'none'
    max_itr = 1500
    patience = 1500
    lr_stage2 = 1.0e-4

    model.quantizer_type = quantizer_type
    model.quantizer_noise_type = quantizer_noise_type
    model.soft_round_temperature = 1e-4
    model.noise_parameter = 1.0
    encoder_logs_best = _hardround_test(model, coords, target_pixels, total_pixels, args.lambda_rate, criterion)
    best_state = model.get_param()

    optimizer = torch.optim.Adam(all_parameters, lr=lr_stage2)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(max_itr / args.freq_valid, 1), eta_min=1e-5)

    print(f'\n===== STAGE 2: {max_itr} steps (ste, lr={lr_stage2}, patience={patience}) =====')
    print(f'{"itr":>7} {"loss e-3":>10} {"psnr_db":>9} {"lat_bpp":>8} {"nn_bpp":>8} {"lr":>10}  record')

    model.train()
    cnt_record = 0
    for cnt in range(max_itr):
        if cnt - cnt_record > patience:
            model.set_param(best_state)
            current_lr = scheduler.get_last_lr()[0]
            for g in optimizer.param_groups:
                g['lr'] = current_lr
            cnt_record = cnt

        for p in all_parameters:
            p.grad = None

        recon_flat, rate_pr, _ = model(coords)
        mse = criterion(recon_flat, target_pixels)
        bpp = rate_pr.sum() / total_pixels
        loss = mse + args.lambda_rate * bpp
        loss.backward()
        nn.utils.clip_grad_norm_(all_parameters, 0.1, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        if (cnt + 1) % args.freq_valid == 0 or (cnt + 1) == max_itr:
            encoder_logs = _hardround_test(model, coords, target_pixels, total_pixels, args.lambda_rate, criterion)
            record = ''
            if encoder_logs['loss'] < encoder_logs_best['loss']:
                best_state = model.get_param()
                encoder_logs_best = encoder_logs
                cnt_record = cnt
                record = '  record'
            cur_lr = scheduler.get_last_lr()[0]
            print(f'{cnt + 1:7d} {encoder_logs["loss"] * 1e3:10.4f} {encoder_logs["psnr"]:9.3f} '
                  f'{encoder_logs["lat_bpp"]:8.4f} {encoder_logs["nn_bpp"]:8.4f} {cur_lr:10.2e}{record}')
            scheduler.step()
            model.train()

    model.set_param(best_state)
    print(f'Reloaded best STAGE 2 model (loss={encoder_logs_best["loss"]:.6f}, '
          f'psnr={encoder_logs_best["psnr"]:.3f}dB).')

    if recon_path is not None:
        with torch.no_grad():
            model.eval()
            recon_flat, _, _ = model(coords)
            model.train()
        recon_img_s2 = recon_flat.view(1, H, W, 3).permute(0, 3, 1, 2).clamp(0, 1)
        os.makedirs(os.path.dirname(recon_path), exist_ok=True)
        vutils.save_image(recon_img_s2, recon_path)

    # ============================================================
    # NN weight quantization 
    # ============================================================
    print('\n===== NN QUANTIZATION (greedy q_step / expgol_cnt search) =====')
    model = quantize_model(model, None, coords, target_pixels, args)

    # ============================================================
    # Post-training refinement of quantized NN 
    # ============================================================
    print('\n===== RDOQ (per-parameter +/- shift search) =====')
    model = rdoq_model(model, coords, target_pixels, args)
    # ============================================================
    # Final evaluation with true hardround
    # ============================================================
    model.eval()
    with torch.no_grad():
        recon_flat, rate_pr, _ = model(coords)
        rate_latent_bits = rate_pr.sum().item()
        bpp_latent = rate_latent_bits / total_pixels

        rate_mlp, rate_arm, rate_conv = compute_model_rate(model)
        rate_nn_bits = float(rate_mlp)
        bpp_nn = rate_nn_bits / total_pixels
        rate_per_module = model.get_network_rate()

        bpp_contour = float(contour_bits) / total_pixels
        bpp_final = bpp_latent + bpp_nn + bpp_contour
        recon_img = recon_flat.view(1, H, W, 3).permute(0, 3, 1, 2).clamp(0, 1)
        psnr_v, ssim_v = compute_psnr_ssim(img_tensor, recon_img)

    print('\n========= FINAL (hardround eval) =========')
    print(f'  lambda  = {args.lambda_rate}')
    print(f'  bpp_lat = {bpp_latent:.4f}')
    print(f'  bpp_nn  = {bpp_nn:.4f}  (NN bits = {rate_nn_bits:.1f})')
    print(f'  bpp_cnt = {bpp_contour:.4f}  (contour bits = {int(contour_bits)})')
    print(f'  bpp     = {bpp_final:.4f}')
    print(f'  PSNR    = {psnr_v:.3f} dB')
    print(f'  SSIM    = {ssim_v:.4f}')
    print('  NN rate per module (bits):')
    for _mname, _mrate in rate_per_module.items():
        _w = float(_mrate.get('weight', 0.0))
        _b = float(_mrate.get('bias', 0.0))
        print(f'    {_mname:<14s}  weight={_w:>10.1f}  bias={_b:>8.1f}  total={_w+_b:>10.1f}')

    result = {
        'psnr': psnr_v,
        'ssim': ssim_v,
        'bpp': bpp_final,
        'bpp_lat': bpp_latent,
        'bpp_nn': bpp_nn,
        'bpp_contour': bpp_contour,
        'contour_bits': int(contour_bits),
        'rate_latent_bits': rate_latent_bits,
        'rate_nn_bits': rate_nn_bits,
        'rate_arm_bits': float(rate_arm),
        'rate_conv_bits': float(rate_conv),
    }

    # ============================================================
    # Real bitstream: range-code the latents + NN weights, decode
    # them back and reconstruct (controlled by --write_bitstream)
    # ============================================================
    if args.write_bitstream:
        from utils.bitstream import encode_decode_verify
        bitstream_path = saved_path.replace('.pth', '.cool')
        print('\n===== BITSTREAM (range coding + reconstruction) =====')
        bs = encode_decode_verify(model, coords, bitstream_path)

        real_latent_bits = bs['latent_bytes'] * 8
        real_nn_bits = bs['nn_bytes'] * 8
        real_header_bits = bs['header_bytes'] * 8
        # Header is shared side-info (same network structure for every image):
        # kept in the file but NOT counted in the per-image rate.
        real_total_bits = real_latent_bits + real_nn_bits
        real_bpp = (real_total_bits + float(contour_bits)) / total_pixels

        recon_dec = bs['recon'].clamp(0, 1)
        psnr_dec, ssim_dec = compute_psnr_ssim(img_tensor, recon_dec)

        print(f'  file    = {bitstream_path}  ({bs["total_bytes"]} bytes, '
              f'header {bs["header_bytes"]} B)')
        print(f'  verify  : NN params exact = {bs["nn_match"]}, '
              f'latents exact = {bs["latents_match"]}')
        print(f'  {"":12s} {"estimate":>14s} {"real":>14s}')
        print(f'  {"latent bits":12s} {rate_latent_bits:>14.1f} {real_latent_bits:>14d}')
        print(f'  {"nn bits":12s} {rate_nn_bits:>14.1f} {real_nn_bits:>14d}')
        print(f'  {"header bits":12s} {"-":>14s} {real_header_bits:>14d}   (kept in file, NOT counted in bpp)')
        print(f'  {"bpp":12s} {bpp_final:>14.4f} {real_bpp:>14.4f}   (both incl. contour, excl. header)')
        print(f'  {"PSNR (dB)":12s} {psnr_v:>14.3f} {psnr_dec:>14.3f}')
        print(f'  {"SSIM":12s} {ssim_v:>14.4f} {ssim_dec:>14.4f}')

        recon_dec_path = saved_path.replace('.pth', '_decoded.png')
        vutils.save_image(recon_dec, recon_dec_path)
        print(f'  Saved decoded recon -> {recon_dec_path}')

        result.update({
            'real_bpp': real_bpp,
            'real_psnr': psnr_dec,
            'real_ssim': ssim_dec,
            'real_latent_bits': real_latent_bits,
            'real_nn_bits': real_nn_bits,
            'real_total_bits': real_total_bits,
            'bitstream_path': bitstream_path,
        })

    checkpoint = {'model_state_dict': model.state_dict(), 'binary mask': None}
    torch.save(checkpoint, saved_path)
    print(f'Saved model -> {saved_path}')
    if recon_path is None:
        recon_path = saved_path.replace('.pth', '_recon.png')
    os.makedirs(os.path.dirname(recon_path), exist_ok=True)
    vutils.save_image(recon_img, recon_path)
    print(f'Saved recon -> {recon_path}')

    return result


global args
parser = argparse.ArgumentParser(description='MoRIC + Cool-chic 5.0 Kodak sweep')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-2,
                    help='PresetIntra start_lr: warm-up + phase-1 Adam lr. '
                         'Phase 2 (STE) is hardcoded to 1e-4, matching '
                         'CC5.0.1 PresetIntra.')
parser.add_argument('--data', type=str, default='./dataset/kodak_data_set')

parser.add_argument('--context_arm', type=int, default=20)
parser.add_argument('--dim_arm_mod', type=int, default=20)
parser.add_argument('--mod_base', type=int, default=7)
parser.add_argument('--sythesis_features', type=int, default=18)
parser.add_argument('--local_upsampling_kernel_size', type=int, default=8)
parser.add_argument('--upsampling_kernel_size', type=int, default=8)
parser.add_argument('--upsampling_preconcat_kernel_size', type=int, default=7)

parser.add_argument('--write_bitstream', action='store_true', default=False,
                    help='Encode a real bitstream (range coding), decode + '
                         'reconstruct, and print estimate vs real results.')

parser.add_argument('--use_multi_region', dest='use_multi_region',
                    action='store_true', default=True,
                    help='Derive K regions per image from the lossy mask '
                         '(default).')
parser.add_argument('--no_multi_region', dest='use_multi_region',
                    action='store_false',
                    help='Single-region mode: K=1, no mask / contour bits.')
parser.add_argument('--mask_dir', type=str,
                    default='./dataset/kodak_data_set/kodak_mask',
                    help='Merged binary masks (kodim{NN}.png) from the '
                         'DeepLab+SAM region pipeline.')
parser.add_argument('--lossy_mask_dir', type=str,
                    default='./dataset/kodak_data_set/kodak_lossy_mask',
                    help='Where lossy contour-encoded masks (kodim{NN}.png '
                         '+ kodim{NN}_bits.txt) are cached/read.')
parser.add_argument('--regen_lossy', action='store_true', default=False,
                    help='Force re-encoding lossy masks even if cached.')

parser.add_argument('--stage1_steps', type=int, default=100000,
                    help='STAGE 1 (main) iteration budget (upstream --n_itr).')
parser.add_argument('--print_every', type=int, default=100)

parser.add_argument('--freq_valid', type=int, default=100,
                    help='Iterations between hardround-validation best-model '
                         'checks in STAGE 1 / STAGE 2 (PresetIntra: 100).')

parser.add_argument('--use_candidate', action='store_true', default=False,
                    help='Run CC5.0.1 PresetIntra warmup (7->4 candidates x '
                         '400 iters) and use the winner as the main-stage '
                         'init.')

parser.add_argument('--lambda_rate', type=float, default=4e-3)
parser.add_argument(
    '--lambda_rate_list',
    type=float,
    nargs='+',
    default=[4e-3],
    help='list of lambda weights',
)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=24)
parser.add_argument('--out_dir', type=str, default='./compress_out')

args = parser.parse_args()

# ============================================================
# Result accumulators (per-lambda lists of per-image values)
# ============================================================
all_psnr_list_of_lists = []
all_ssim_list_of_lists = []
all_bpp_list_of_lists = []
all_bpp_lat_list_of_lists = []
all_bpp_nn_list_of_lists = []
all_rate_lat_bits_list_of_lists = []
all_rate_nn_bits_list_of_lists = []
all_rate_arm_bits_list_of_lists = []
all_rate_conv_bits_list_of_lists = []
all_real_psnr_list_of_lists = []
all_real_bpp_list_of_lists = []

for num, lambda_rate in enumerate(args.lambda_rate_list):
    seed_everything(1)
    all_psnr = []
    all_ssim = []
    all_bpp = []
    all_bpp_lat = []
    all_bpp_nn = []
    all_rate_lat_bits = []
    all_rate_nn_bits = []
    all_rate_arm_bits = []
    all_rate_conv_bits = []
    all_real_psnr = []
    all_real_bpp = []

    args.lambda_rate = lambda_rate

    for it in range(args.start_index, args.end_index):
        idx_str = f"{it + 1:02d}"
        image_path = f'{args.data}/kodim{idx_str}/data/kodim{idx_str}.png'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        img = Image.open(image_path).convert('RGB')
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        _, _, H, W = img_tensor.shape
        print(f'Loaded {image_path}  ->  {W}x{H} ({H * W} pixels)')

        if args.use_multi_region:
            lossy_mask_path, contour_bits = _get_lossy_mask(
                args.mask_dir, args.lossy_mask_dir, idx_str, regen=args.regen_lossy
            )
            region_map, K = _region_map_from_mask(lossy_mask_path, H, W)
        else:
            region_map, K = torch.zeros((H, W), dtype=torch.long), 1
            contour_bits = 0
            print('[mask] multi-region disabled: K=1, no contour bits')
        args.num_regions = K
        print(f'Regions: K={K}  (per-region pixel counts: '
              f'{[int((region_map == k).sum().item()) for k in range(K)]})')

        min_div = 2 ** (args.mod_base - 1)
        if H % min_div != 0 or W % min_div != 0:
            print(f'WARNING: H={H} or W={W} not divisible by {min_div} (=2^(mod_base-1)). '
                  f'Consider adjusting --mod_base.')

        coords = get_mgrid(W, H).to(device)

        # Per-args fields needed by quantize_model and prints
        args.patch_h = H
        args.patch_w = W
        args.all_pix_num = H * W
        args.eval_pix_num = H * W
        args.scale = 1
        args.hidden_features = 24
        args.hidden_layer = 0
        args.sparsity = 0.0
        args.mod_hid_layer = 0
        print(args)

        folder_path = './saved/modbase_' + str(args.mod_base) + '/context_' + str(args.context_arm) + '_arm_mod_' + str(args.dim_arm_mod)
        make_path(folder_path)
        folder_path_ = folder_path + '/cc5_K' + str(args.num_regions)
        make_path(folder_path_)
        saved_path = (
            folder_path_ + '/inr_mod_' + str(args.dim_arm_mod)
            + '_KODAK_' + str(args.sythesis_features)
            + '_pw_' + str(args.lambda_rate)
            + '_img' + str(it) + '.pth'
        )

        lam_str = _format_lambda(args.lambda_rate)
        results_folder = './results/lambda_' + lam_str + '/k' + str(args.num_regions)
        make_path(results_folder)
        recon_path = (
            results_folder + '/context_' + str(args.context_arm)
            + '_arm' + str(args.dim_arm_mod)
            + '_sys' + str(args.sythesis_features)
            + '_kodim' + idx_str + '.png'
        )

        model = build_model(region_map, args, device)
        print(model)
        print('train the', it, '-th image')

        if args.use_candidate:
            best_state, warmup_info = train_with_candidates(
                build_model_fn=lambda: build_model(region_map, args, device),
                coords=coords,
                target_pixels=img_tensor.permute(0, 2, 3, 1).reshape(1, -1, 3),
                total_pixels=H * W,
                lambda_rate=args.lambda_rate,
                lr=args.lr,
            )
            model.set_param(best_state)
            print(f'[warmup] loaded best candidate state '
                  f'(winner_id={warmup_info["winner_id"]}) into main model.')

        out = train(model, img_tensor, coords, args, it, saved_path, device,
                    contour_bits=contour_bits, recon_path=recon_path)

        all_psnr.append(out['psnr'])
        all_ssim.append(out['ssim'])
        all_bpp.append(out['bpp'])
        all_bpp_lat.append(out['bpp_lat'])
        all_bpp_nn.append(out['bpp_nn'])
        all_rate_lat_bits.append(out['rate_latent_bits'])
        all_rate_nn_bits.append(out['rate_nn_bits'])
        all_rate_arm_bits.append(out['rate_arm_bits'])
        all_rate_conv_bits.append(out['rate_conv_bits'])

        print('Trained the image. PSNR:', out['psnr'],
              ' SSIM:', out['ssim'],
              ' bpp:', out['bpp'],
              ' (lat:', out['bpp_lat'],
              ' nn:', out['bpp_nn'],
              ' cnt:', out['bpp_contour'], ')')
        print('Running PSNR list:', all_psnr)
        print('Running bpp  list:', all_bpp)
        if 'real_psnr' in out:
            all_real_psnr.append(out['real_psnr'])
            all_real_bpp.append(out['real_bpp'])
            print('Running REAL PSNR list:', all_real_psnr)
            print('Running REAL bpp  list:', all_real_bpp)
        print('Current Ave PSNR:', np.mean(all_psnr), 'Ave bpp:', np.mean(all_bpp))
        if all_real_psnr:
            print('Current Ave REAL PSNR:', np.mean(all_real_psnr),
                  'Ave REAL bpp:', np.mean(all_real_bpp))

        torch.cuda.empty_cache()

    all_psnr_list_of_lists.append(all_psnr)
    all_ssim_list_of_lists.append(all_ssim)
    all_bpp_list_of_lists.append(all_bpp)
    all_bpp_lat_list_of_lists.append(all_bpp_lat)
    all_bpp_nn_list_of_lists.append(all_bpp_nn)
    all_rate_lat_bits_list_of_lists.append(all_rate_lat_bits)
    all_rate_nn_bits_list_of_lists.append(all_rate_nn_bits)
    all_rate_arm_bits_list_of_lists.append(all_rate_arm_bits)
    all_rate_conv_bits_list_of_lists.append(all_rate_conv_bits)
    all_real_psnr_list_of_lists.append(all_real_psnr)
    all_real_bpp_list_of_lists.append(all_real_bpp)

    print('.......Complete all dataset training for lambda', lambda_rate, '......')
    print('Ave PSNR :', np.mean(all_psnr))
    print('Ave SSIM :', np.mean(all_ssim))
    print('Ave bpp  :', np.mean(all_bpp))
    print('Ave bpp_lat:', np.mean(all_bpp_lat))
    print('Ave bpp_nn :', np.mean(all_bpp_nn))
    if all_real_psnr:
        print('Ave REAL PSNR:', np.mean(all_real_psnr))
        print('Ave REAL bpp :', np.mean(all_real_bpp))

print("======== ALL Results ========")
for i, lambda_rate in enumerate(args.lambda_rate_list):
    print(f"Lambda = {lambda_rate}:")
    print("  all_psnr:", all_psnr_list_of_lists[i])
    print("  all_ssim:", all_ssim_list_of_lists[i])
    print("  all_bpp:", all_bpp_list_of_lists[i])
    print("  all_bpp_lat:", all_bpp_lat_list_of_lists[i])
    print("  all_bpp_nn:", all_bpp_nn_list_of_lists[i])
    print("  all_rate_lat_bits:", all_rate_lat_bits_list_of_lists[i])
    print("  all_rate_nn_bits:", all_rate_nn_bits_list_of_lists[i])
    print("  all_rate_arm_bits:", all_rate_arm_bits_list_of_lists[i])
    print("  all_rate_conv_bits:", all_rate_conv_bits_list_of_lists[i])
    if all_real_psnr_list_of_lists[i]:
        print("  all_real_psnr:", all_real_psnr_list_of_lists[i])
        print("  all_real_bpp:", all_real_bpp_list_of_lists[i])
    print(f"  Ave PSNR: {np.mean(all_psnr_list_of_lists[i]):.4f}")
    print(f"  Ave SSIM: {np.mean(all_ssim_list_of_lists[i]):.4f}")
    print(f"  Ave bpp : {np.mean(all_bpp_list_of_lists[i]):.4f}")
    if all_real_psnr_list_of_lists[i]:
        print(f"  Ave REAL PSNR: {np.mean(all_real_psnr_list_of_lists[i]):.4f}")
        print(f"  Ave REAL bpp : {np.mean(all_real_bpp_list_of_lists[i]):.4f}")
    print("---------------------------------")
