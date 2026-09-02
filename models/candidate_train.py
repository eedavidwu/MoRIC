import math

import torch
from torch import nn


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


def _train_candidate(model, coords, target_pixels, total_pixels, lambda_rate, lr, n_iter, criterion):
    model.quantizer_type = 'softround'
    model.quantizer_noise_type = 'kumaraswamy'
    model.soft_round_temperature = 0.3
    model.noise_parameter = 2.0

    all_parameters = list(model.parameters())
    optimizer = torch.optim.Adam(all_parameters, lr=lr)

    model.train()
    for step in range(n_iter):
        for p in all_parameters:
            p.grad = None
        recon_flat, rate_pr, _ = model(coords)
        mse = criterion(recon_flat, target_pixels)
        bpp = rate_pr.sum() / total_pixels
        loss = mse + lambda_rate * bpp
        loss.backward()
        nn.utils.clip_grad_norm_(all_parameters, 0.1, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

    return _hardround_test(model, coords, target_pixels, total_pixels, lambda_rate, criterion)


def train_with_candidates(build_model_fn, coords, target_pixels, total_pixels,
                           lambda_rate, lr, num_candidates=7, top_k=4, n_iter=400):
    criterion = nn.MSELoss()

    print(f'\n{"-" * 30}  Warm-up stage 0: {num_candidates} candidates x {n_iter} iters  {"-" * 30}')
    candidates = []
    for cid in range(num_candidates):
        print(f'\nCandidate n. {cid:<2}\n-------------------------')
        model = build_model_fn()
        metrics = _train_candidate(model, coords, target_pixels, total_pixels,
                                    lambda_rate, lr, n_iter, criterion)
        candidates.append({'id': cid, 'state': model.get_param(), 'metrics': metrics})
        print(f'  loss={metrics["loss"] * 1e3:.4f}e-3  psnr={metrics["psnr"]:.3f}dB  '
              f'lat_bpp={metrics["lat_bpp"]:.4f}')
        del model
        torch.cuda.empty_cache()

    candidates.sort(key=lambda c: c['metrics']['loss'])
    print(f'[stage 0] ranked ids = {[c["id"] for c in candidates]}')

    top = candidates[:top_k]
    print(f'\n{"-" * 30}  Warm-up stage 1: top-{top_k} x {n_iter} iters  {"-" * 30}')
    for c in top:
        print(f'\nCandidate ID = {c["id"]:<2} (refine)\n-------------------------')
        model = build_model_fn()
        model.set_param(c['state'])
        metrics = _train_candidate(model, coords, target_pixels, total_pixels,
                                    lambda_rate, lr, n_iter, criterion)
        c['state'] = model.get_param()
        c['metrics'] = metrics
        print(f'  loss={metrics["loss"] * 1e3:.4f}e-3  psnr={metrics["psnr"]:.3f}dB  '
              f'lat_bpp={metrics["lat_bpp"]:.4f}')
        del model
        torch.cuda.empty_cache()

    best = min(top, key=lambda c: c['metrics']['loss'])
    print(f'\n>>> [warmup BEST] id={best["id"]}  loss={best["metrics"]["loss"]:.6f}  '
          f'psnr={best["metrics"]["psnr"]:.3f}dB\n')

    info = {
        'stage0': [(c['id'], c['metrics']['loss']) for c in candidates],
        'stage1': [(c['id'], c['metrics']['loss']) for c in top],
        'winner_id': best['id'],
    }
    return best['state'], info
