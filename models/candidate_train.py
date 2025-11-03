
from models.model import Masked_INR
import copy
import torch
from torch import nn
import numpy as np
from torch import  nn
import torchvision.utils as vutils




def train_with_candidates(
    args, target_mask,target_mask_flat, dataloader,
    steps_stage1=400, steps_stage2=400,
    num_candidates=7, top_k=3, it=0, saved_path="./"
):
    candidates = []

    # === Stage 1: Train initial candidates ===
    for cand_id in range(num_candidates):
        print(f"\n🔁 Stage-1 Training Candidate {cand_id+1}/{num_candidates}")
        model = Masked_INR(args,target_mask,sparsity=args.sparsity,in_features=2, out_features=3*args.scale*args.scale, hidden_features=args.hidden_features, hidden_layers=args.hidden_layer).cuda()

        psnr, rate, loss, best_model, = candidate_train(args,
            target_mask_flat, model, dataloader,
            steps_stage1,
            steps_til_summary=steps_stage1
        )

        candidates.append({
            'model': best_model,
            'psnr': psnr,
            'rate': rate,
            'loss': loss,
        })

    # === Select Top-K candidates ===
    candidates = sorted(candidates, key=lambda x: x['loss'], reverse=False)
    top_candidates = candidates[:top_k]
    print(f"\n🌟 Selected Top-{top_k} Candidates by RD cost")

    # === Stage 2: Refine top candidates ===
    final_candidates = []
    for cand_id, cand in enumerate(top_candidates):
        print(f"\n🎯 Stage-2 Refining Candidate {cand_id+1}/{top_k}")
        model = cand['model']

        psnr, rate, loss, best_model = candidate_train(args,
            target_mask_flat, model, dataloader,
            steps_stage2,
            steps_til_summary=steps_stage2
        )
        final_candidates.append({
            'model': best_model,
            'psnr': psnr,
            'rate': rate,
            'loss': loss,
        })

    # === Final Selection ===
    best = min(final_candidates, key=lambda x: x['loss'])  # or RD-score
    print(f"\n🏆 Best Candidate Final PSNR: {best['psnr']:.2f}, Rate: {best['rate']:.4f}")

    return best['model']


def candidate_train(args,target_mask, model,dataloader, total_steps,steps_til_summary):
    vis_colum=3
    best_psnr=0
    criterion = nn.MSELoss().cuda()
    base_params = [p for name, p in model.named_parameters()]
    optim = torch.optim.Adam([{'params': base_params, 'lr': args.lr}])
    
    for batch_idx, (img_in,_) in enumerate(dataloader, 0):
        model.train()
        vis_colum=1
        batch_size,_,height,width=img_in.shape
        pixels = img_in.permute(0, 2, 3, 1).view(batch_size,-1, 3).cuda()
        
       
        coords = get_mgrid(width//args.scale,height//args.scale, 2).cuda()
        losses = []
        
        best_psnr=0
        best_rate=0
        
        initial_noise_param = 2.0
        final_noise_param = 2.0
        initial_temperature = 0.3
        final_temperature = 0.3
        print('start temperature:', initial_temperature,'noise parameter:',initial_noise_param)
        print('end temperature:', final_temperature,'noise parameter:',final_noise_param)

        print("********************Start from stage I")
       
        best_rd=1000
        patience = 100000

        for step in range(total_steps+1):
            ###stage 1:
            model.train()
            
            model.noise_parameter = initial_noise_param - (step / total_steps) * (initial_noise_param - final_noise_param)
            model.soft_round_temperature = initial_temperature - (step / total_steps) * (initial_temperature - final_temperature)
           
            model_output,rate,_ = model(coords)   
          
            bits_rate=rate.sum()/(args.all_pix_num)
            loss_mse=criterion(model_output, pixels)
        
            loss=args.lambda_rate*bits_rate+loss_mse
            losses.append(loss.item())
            if not step % steps_til_summary or (step==total_steps):
                psnr_this_iter=loss_to_psnr(loss_mse.item())
                
                if (loss<best_rd) and (step>0):
                    best_rate = bits_rate.item()
                    best_psnr=psnr_this_iter
                    best_rd=loss
                    best_model = model.get_param()
                    print("Step %d, BEST PSNR: %0.6f, Total loss %0.6f" % (step, psnr_this_iter,loss),'with its rate', bits_rate.item(), 'latent_bits', rate.sum().item())
                    
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10,norm_type=2.0, error_if_nonfinite=False)
          
            optim.step()
    model.set_param(best_model)
        
    return best_psnr, best_rate, best_rd, model


def get_mgrid(w_sidelen,h_sidelen, dim=2):
    
    x = torch.linspace(-1, 1, steps=w_sidelen) 
    y = torch.linspace(-1, 1, steps=h_sidelen)  
    tensors = (x, y) if dim == 2 else (x, ) * dim  
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)  
    mgrid = mgrid.unsqueeze(0).permute(0,3,2,1)

    return mgrid

def loss_to_psnr(loss, max=1):
  return 10*np.log10(max**2/np.asarray(loss))
