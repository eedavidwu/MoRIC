
import time
from collections import OrderedDict

import torch
from torch import nn

from enc.training.loss import loss_function
from enc.utils.misc import get_q_step_from_parameter_name


def _flatten_subset(param, weight_or_bias):
    
    subset = OrderedDict(
        (k, v) for k, v in param.items() if f".{weight_or_bias}" in k
    )
    if len(subset) == 0:
        return None, None
    flat = torch.cat([v.flatten() for v in subset.values()])
    sizes = OrderedDict((k, tuple(v.shape)) for k, v in subset.items())
    return flat, sizes


def _unflatten_into(full_param, sizes, flat):
    
    out = OrderedDict(full_param)
    ptr = 0
    for k, sz in sizes.items():
        n = 1
        for s in sz:
            n *= int(s)
        out[k] = flat[ptr:ptr + n].view(sz) if len(sz) > 0 else flat[ptr:ptr + 1].view(())
        ptr += n
    return out


@torch.no_grad()
def _score_model(model, coords, target_pixels, lmbda):
   
    recon_flat, rate_per_region, _ = model(coords)
    rate_latent_bits = rate_per_region.sum()

    rate_nn_bits = 0.0
    rate_per_module = model.get_network_rate()
    for _, module_rate in rate_per_module.items():
        for _, param_rate in module_rate.items():
            rate_nn_bits = rate_nn_bits + float(param_rate)

    out = loss_function(
        recon_flat,
        rate_per_region,
        target_pixels,
        lmbda=lmbda,
        rate_mlp_bit=rate_nn_bits,
        compute_logs=False,
    )
    return out.loss


@torch.no_grad()
def rdoq_model(model, coords, target_pixels, args, verbosity=0):
   
    start_time = time.time()
    model.eval()
    lmbda = float(args.lambda_rate)

    print("Start discrete RDOQ on neural network parameters...")
    initial_loss = float(_score_model(model, coords, target_pixels, lmbda))
    print(f"  Initial RD loss = {initial_loss:.6f}")

    test_cnt = 0
    module_to_rdoq = {
        module_name: getattr(model, module_name)
        for module_name in model.modules_to_send
    }

    for module_name, cur_module in sorted(module_to_rdoq.items()):
        q_step_dict = model.nn_q_step[module_name]

        # Skip modules with no q_step set (e.g. IFCE when use_ifce=False)
        if q_step_dict is None or q_step_dict.get("weight") is None:
            continue

        for weight_or_bias in ["weight", "bias"]:
            full_param = cur_module.get_param()
            flat_param, sizes = _flatten_subset(full_param, weight_or_bias)

            # Module may have no parameters of this kind (e.g. bias-less layers)
            if flat_param is None or flat_param.numel() == 0:
                continue

            q_step = float(get_q_step_from_parameter_name(
                f".{weight_or_bias}", q_step_dict
            ))

            best_loss = float(_score_model(model, coords, target_pixels, lmbda))
            random_idx = torch.randperm(flat_param.numel())
            n_hits = 0

            for idx_val in random_idx:
                initial_value = float(flat_param[idx_val])
                best_value = initial_value

                for sign in [-1, 1]:
                    for shift in range(1, 16):
                        flat_param[idx_val] = initial_value + sign * shift * q_step

                        new_full = _unflatten_into(full_param, sizes, flat_param)
                        cur_module.set_param(new_full)

                        candidate_loss = float(
                            _score_model(model, coords, target_pixels, lmbda)
                        )
                        test_cnt += 1

                        if candidate_loss < best_loss:
                            best_loss = candidate_loss
                            best_value = float(flat_param[idx_val])
                            n_hits += 1
                            if verbosity > 0:
                                print(
                                    f"  hit  module={module_name:12s} "
                                    f"w_or_b={weight_or_bias:6s} "
                                    f"shift={sign * shift:>4d}  "
                                    f"loss={best_loss:.6f}"
                                )
                        else:
                            
                            break

                flat_param[idx_val] = best_value
              
                new_full = _unflatten_into(full_param, sizes, flat_param)
                cur_module.set_param(new_full)
                
                full_param = cur_module.get_param()
                flat_param, sizes = _flatten_subset(full_param, weight_or_bias)

            print(
                f"  RDOQ  module={module_name:12s} "
                f"w_or_b={weight_or_bias:6s} "
                f"n_param={flat_param.numel():6d}  "
                f"n_hits={n_hits:6d}  "
                f"loss={best_loss:.6f}"
            )

    elapsed = time.time() - start_time
    final_loss = float(_score_model(model, coords, target_pixels, lmbda))
    print(
        f"RDOQ done: loss {initial_loss:.6f} -> {final_loss:.6f} "
        f"(delta = {final_loss - initial_loss:+.6f}), "
        f"{test_cnt} candidate forwards in {elapsed:.1f}s\n"
    )

    return model
