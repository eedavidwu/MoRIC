
import itertools
import time
from typing import Optional, OrderedDict
import numpy as np
import torch
from torch import nn
from enc.utils.misc import exp_golomb_nbins
from enc.training.loss import loss_function

from enc.utils.misc import (
    MAX_AC_MAX_VAL,
    POSSIBLE_EXP_GOL_COUNT,
    POSSIBLE_Q_STEP,
    DescriptorNN,
    get_q_step_from_parameter_name,
)
from torch import Tensor


def _quantize_parameters(
    fp_param: OrderedDict[str, Tensor],
    q_step: DescriptorNN,
) -> Optional[OrderedDict[str, Tensor]]:
   
    q_param = OrderedDict()
    for k, v in fp_param.items():
        current_q_step = get_q_step_from_parameter_name(k, q_step)
        sent_param = torch.round(v / current_q_step)

        if sent_param.abs().max() > MAX_AC_MAX_VAL:
           
            return None

        q_param[k] = sent_param * current_q_step

    return q_param
    
def loss_to_psnr(loss, max=1):
  return 10*np.log10(max**2/np.asarray(loss))

@torch.no_grad()
def quantize_model(frame_encoder,binary_mask,coords,frame_gt,args):
    
    frame_encoder.eval()
    mse_loss = nn.MSELoss().cuda()


  
    module_to_quantize = {
        module_name: getattr(frame_encoder, module_name)
        for module_name in frame_encoder.modules_to_send
    }

    for module_name, cur_module in sorted(module_to_quantize.items()):
        
        best_loss = 1e6

      
        all_q_step = POSSIBLE_Q_STEP.get(module_name)
        all_expgol_cnt = POSSIBLE_EXP_GOL_COUNT.get(module_name)

       
        fp_param = cur_module.get_param()

        best_q_step = {}
        
        final_best_expgol_cnt = {}
        for q_step_w, q_step_b in itertools.product(all_q_step.get("weight"), all_q_step.get("bias")):
        
            current_q_step: DescriptorNN = {"weight": q_step_w, "bias": q_step_b}

           
            q_param = _quantize_parameters(fp_param, current_q_step)

          
            if q_param is None:
                continue

            cur_module.set_param(q_param)

           
            setattr(frame_encoder, module_name, cur_module)

            frame_encoder.nn_q_step[module_name] = current_q_step

          
            decode_out,decode_rate,_ = frame_encoder.forward(coords,binary_mask)
            loss_mse=mse_loss(decode_out,frame_gt)

            psnr_eval=loss_to_psnr(loss_mse.item())
            computed_rate=decode_rate.sum()/(coords.shape[1])

            param = cur_module.get_param()

          
            best_expgol_cnt = {}
            for weight_or_bias in ["weight", "bias"]:

                
                cur_best_expgol_cnt = None
                
                cur_best_rate = 1e9

                sent_param = []
                for parameter_name, parameter_value in param.items():

                   
                    current_sent_param = (parameter_value / current_q_step.get(weight_or_bias)).view(-1)

                    if parameter_name.endswith(weight_or_bias):
                        sent_param.append(current_sent_param)

               
                v = torch.cat(sent_param)

                for expgol_cnt in all_expgol_cnt.get(weight_or_bias):
                    cur_rate = exp_golomb_nbins(v, count=expgol_cnt)
                    if cur_rate < cur_best_rate:
                        cur_best_rate = cur_rate
                        cur_best_expgol_cnt = expgol_cnt

                best_expgol_cnt[weight_or_bias] = int(cur_best_expgol_cnt)

            frame_encoder.nn_expgol_cnt[module_name] = best_expgol_cnt

            rate_mlp = 0.0
            rate_per_module = frame_encoder.get_network_rate()
            for _, module_rate in rate_per_module.items():
                for _, param_rate in module_rate.items(): 
                    rate_mlp += param_rate

            loss_fn_output = loss_function(
                decode_out,
                decode_rate,
                frame_gt,
                lmbda=args.lambda_rate,
                rate_mlp_bit=rate_mlp,
                compute_logs=True,
            )


            
            if loss_fn_output.loss < best_loss:
                best_loss = loss_fn_output.loss
                best_q_step = current_q_step
                final_best_expgol_cnt = best_expgol_cnt
              

        frame_encoder.nn_q_step[module_name] = best_q_step
        frame_encoder.nn_expgol_cnt[module_name] = final_best_expgol_cnt

        q_param = _quantize_parameters(fp_param, frame_encoder.nn_q_step[module_name])
        assert q_param is not None, (
            "_quantize_parameters() failed with q_step "
            f"{frame_encoder.nn_q_step[module_name]}"
        )

        cur_module.set_param(q_param)
       
        setattr(frame_encoder, module_name, cur_module)

    return frame_encoder
