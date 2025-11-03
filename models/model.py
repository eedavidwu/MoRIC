
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import torch.autograd as autograd
from torch import Tensor, index_select, nn

from utils.quantizer import quantize
from utils.arm import (
    Arm,
    _get_neighbor,
    _get_non_zero_pixel_ctx_index,
    _laplace_cdf,
)
from utils.upsampling import Upsampling

from enc.utils.misc import (
    DescriptorCoolChic,
    DescriptorNN,
    measure_expgolomb_rate,
)
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, TypedDict
from itertools import islice
class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)
        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


       
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
       
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

       
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        return out

    @staticmethod
    def backward(ctx, g):
       
        return g, None

class GetSubnet_batch(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        
        out = scores.clone()
        
        batch_size,w1,w2=scores.shape
        score_reshape=scores.view(batch_size,-1)
        _, indices = torch.sort(score_reshape, dim=1, descending=True)
       
        j = int((1 - k) * score_reshape.size(1))

        binary_mask = torch.zeros_like(score_reshape)
        binary_mask.scatter_(1, indices[:, :j], 1)
        binary_mask = binary_mask.view(batch_size, w1, w2)
        return binary_mask

    @staticmethod
    def backward(ctx, g):
       
        return g, None

class NonAffineBatchNorm(nn.BatchNorm1d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)


class SynthesisLayer(nn.Module):
    def __init__(
        self,
        input_ft: int,
        output_ft: int,
        kernel_size: int,
        non_linearity: nn.Module = nn.Identity()
    ):
       
        super().__init__()

        self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        self.conv_layer = nn.Conv2d(
            input_ft,
            output_ft,
            kernel_size
        )

        self.non_linearity = non_linearity

       
        with torch.no_grad():
            self.conv_layer.weight.data = self.conv_layer.weight.data / output_ft ** 2
            self.conv_layer.bias.data = self.conv_layer.bias.data * 0.

    def forward(self, x: Tensor) -> Tensor:
        return self.non_linearity(self.conv_layer(self.pad(x)))

class SynthesisResidualLayer(nn.Module):
    def __init__(
        self,
        input_ft: int,
        output_ft: int,
        kernel_size: int,
        non_linearity: nn.Module = nn.Identity()
    ):
       
        super().__init__()

        assert input_ft == output_ft,\
            f'Residual layer in/out dim must match. Input = {input_ft}, output = {output_ft}'

        self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        self.conv_layer = nn.Conv2d(
            input_ft,
            output_ft,
            kernel_size
        )

        self.non_linearity = non_linearity

        with torch.no_grad():
            self.conv_layer.weight.data = self.conv_layer.weight.data * 0.
            self.conv_layer.bias.data = self.conv_layer.bias.data * 0.

    def forward(self, x: Tensor) -> Tensor:
        return self.non_linearity(self.conv_layer(self.pad(x)) + x)

class ModConv(nn.Module):
    def __init__(
        self,
        in_channels,
        hid_channels,
        out_channels,
        mod_layer,
    ):
        super().__init__()
        self.residual = False
        self.hid_channels=hid_channels
        self.hid_layer=mod_layer
       

        self.conv1_1 = SynthesisLayer(in_channels,hid_channels,1,nn.GELU())
        self.conv1_2 = SynthesisLayer(hid_channels,3,1,nn.GELU())
        self.conv2_1 = SynthesisResidualLayer(3,3,3,nn.GELU())
        self.conv2_2 = SynthesisResidualLayer(3,3,3)

    def get_param(self) -> OrderedDict[str, Tensor]:
        
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})
    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        
        self.load_state_dict(param)

    def forward(self, x):
        
        out_0=self.conv1_1(x) 
        out_1=self.conv1_2(out_0)
      
        out_2=self.conv2_1(out_1)
        out_3=self.conv2_2(out_2)
       
        return out_3
class LocallyConnectedBlock(nn.Module):
    def __init__(self, in_channels, global_hid_channels, local_hid_channels, out_channels, mod_layer):
        super().__init__()
        self.net = []
        self.net.append(nn.Sequential(
           SynthesisLayer(2,local_hid_channels,1,nn.GELU())
        ))
        self.net.append(nn.Sequential(
           SynthesisResidualLayer(local_hid_channels,local_hid_channels,1,nn.GELU())
        ))
        self.net.append(nn.Sequential(
           SynthesisResidualLayer(local_hid_channels,local_hid_channels,1,nn.GELU())
        ))
        self.net.append(nn.Sequential(
           SynthesisResidualLayer(local_hid_channels,3,1)
        ))

        self.net = nn.Sequential(*self.net)

        self.global_net = []
        self.global_net.append(SynthesisLayer(2,local_hid_channels,1,nn.GELU()))
        self.global_net.append(SynthesisResidualLayer(local_hid_channels,local_hid_channels,1,nn.GELU()))
        self.global_net.append(SynthesisResidualLayer(local_hid_channels,local_hid_channels,1,nn.GELU()))
        self.global_net.append(SynthesisResidualLayer(local_hid_channels,3,1))

        self.global_net = nn.Sequential(*self.global_net)
    def get_param(self) -> OrderedDict[str, Tensor]:
        
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})
    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        
        self.load_state_dict(param)
    
    def forward(self, x, y):
        
        output_local = self.net(x)
        output_global= self.global_net(y)
        
        return output_local,output_global
    
class LocalGlobalBlock(LocallyConnectedBlock):
    def __init__(self, in_channels, global_hid_channels,local_hid_channels, out_channels, mod_layer,mask):
        super().__init__(in_channels, global_hid_channels, local_hid_channels, out_channels, mod_layer)
        
        self.mask = mask
        self.agg_func = []

        self.agg_func.append(nn.Sequential(SynthesisLayer(global_hid_channels+6,3,1,nn.GELU())))
        self.agg_func.append(nn.Sequential(SynthesisLayer(global_hid_channels+9,3,1,nn.GELU())))
        self.agg_func.append(nn.Sequential(SynthesisLayer(global_hid_channels+12,3,1,nn.GELU())))
        self.agg_func = nn.Sequential(*self.agg_func)


        self.full_net = []
        self.full_net.append(SynthesisLayer(in_channels,global_hid_channels,1,nn.GELU()))
        self.full_net.append(SynthesisLayer(global_hid_channels,3,1,nn.GELU()))
        self.full_net.append(SynthesisResidualLayer(3,3,3,nn.GELU()))
        self.full_net.append(SynthesisResidualLayer(3,3,3,nn.GELU()))
      

        self.full_net = nn.Sequential(*self.full_net)
    def get_param(self) -> OrderedDict[str, Tensor]:
       
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})
    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
       
        self.load_state_dict(param)
    def forward(self, coordinate,combined_latent):

        coordinate = coordinate
        self.mask = self.mask.bool()
        
        object_latent = torch.zeros_like(coordinate)
        background_latent = torch.zeros_like(coordinate)
        object_latent[self.mask.expand_as(coordinate)] = coordinate[self.mask.expand_as(coordinate)]
        background_latent[~self.mask.expand_as(coordinate)] = coordinate[~self.mask.expand_as(coordinate)]
        
        all_outputs = []
        out_full = []
        
        global_layer_input = background_latent
        local_layer_input = object_latent
        full_layer_input = combined_latent
        id=0
        
        for full_layer in self.full_net:
          full_layer_input = full_layer(full_layer_input)
          all_outputs.append(full_layer_input)
        
        out_full.append(torch.cat(all_outputs[:2], dim=1)) 
        out_full.append(torch.cat(all_outputs[:3], dim=1))
        out_full.append(torch.cat(all_outputs, dim=1))             
        
        
        for local_layer, global_layer, agg_layer in zip(self.net, self.global_net,self.agg_func):
            local_layer_input = local_layer(local_layer_input)
            global_layer_input = global_layer(global_layer_input)
            
            
            if id<3:
                
                device =  out_full[id].device  
                self.mask = self.mask.to(device)  
                self.mask = self.mask.squeeze()
                output_full_local = torch.where(self.mask, out_full[id], torch.zeros_like(out_full[id], device=device))
                output_full_global = torch.where(~self.mask, out_full[id], torch.zeros_like(out_full[id], device=device))
                global_layer_input = agg_layer(torch.cat([global_layer_input,output_full_global],dim=-3))
                local_layer_input = agg_layer(torch.cat([local_layer_input,output_full_local],dim=-3))
                
            id+=1
        global_layer_input = self.global_net[-1](global_layer_input)
        local_layer_input = self.net[-1](local_layer_input)

        return local_layer_input,global_layer_input

class Masked_INR(nn.Module):
    def __init__(self, args,target_mask,sparsity,in_features,  out_features, hidden_features, hidden_layers):
        super().__init__()
        self.sparsity=sparsity
        self.net = []
       
        self.h = target_mask.shape[-2]
        self.w = target_mask.shape[-1]
      
        self.target_mask = target_mask
        self.pe_flag=0
        if self.pe_flag==1:
            self.pe=PosEncodingNeRF(2,(self.h,self.w))
            input_dim=30
        else:
            input_dim=2

       
        self.upsampling_2d = Upsampling(
            args.local_upsampling_kernel_size, args.static_upsampling_kernel,args.highest_flag
        )
        
        self.dim_arm=args.dim_arm_mod
        self.n_hidden_layers_arm=2
        self.arm = Arm(args.context_arm,args.dim_arm_mod, self.n_hidden_layers_arm)
        
       
        self.quantizer_type="softround"
        self.quantizer_noise_type="kumaraswamy"
        self.soft_round_temperature=0.3
        self.noise_parameter=2.0 ##1 means uniform....
        max_mask_size = 9
       
        self.modulation_base_number=args.mod_base
      
        self.fact_shape=[]
        if args.highest_flag==1:
            for i in range (self.modulation_base_number):
                self.fact_shape.append((self.h//(2**i),self.w//(2**i)))
        else:
            for i in range (self.modulation_base_number):
                self.fact_shape.append((self.h//(2**(i+1)),self.w//(2**(i+1))))
        self.fact_shape.reverse()
        max_context_pixel = int((max_mask_size**2 - 1) / 2)
        assert self.dim_arm <= max_context_pixel, (
            f"You can not have more context pixels "
            f" than {max_context_pixel}. Found {self.dim_arm}"
        )
        
        self.mask_size=9
        self.encoder_gains_sf=16
        print('Quantizer parameter: encoding gain ',self.encoder_gains_sf)

        self.all_pix_num=self.h*self.w//args.scale//args.scale
        print('total pixel:',self.all_pix_num)
        
        self.register_buffer(
            "non_zero_pixel_ctx_index",
            _get_non_zero_pixel_ctx_index(args.context_arm),
            persistent=False,
        )
       
        self.latent_factor=args.latent_factor

        self.conv_mod = LocalGlobalBlock(in_channels=self.modulation_base_number, global_hid_channels=args.sythesis_features,local_hid_channels=3, out_channels=hidden_layers+1, mod_layer=args.mod_hid_layer, mask=self.target_mask)

        self.modules_to_send=['arm','conv_mod','upsampling_2d']

        self.nn_q_step: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }
        self.nn_expgol_cnt: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }
        self.modulation_sf= nn.ParameterList()

        self.mask_sf = []
        for layer_idx in range(self.modulation_base_number):
            mod_shape=self.fact_shape[layer_idx]
            shits =  nn.Parameter(torch.zeros(args.batch_size,1,  mod_shape[0], mod_shape[1])).cuda()#.requires_grad=True
            if layer_idx>0:
                masks = F.max_pool2d(target_mask.float(), kernel_size=2)
            else:
                masks = target_mask
            target_mask = masks
            self.mask_sf.append(masks.cuda())
            self.modulation_sf.append(shits)
            print('Get Mod with shape',shits.shape,'at layer:',layer_idx+1)
        

    def quantize_all_latent(self,latent,coords):
        q_shifts_all=[]
  
        q_shifts_all_for_conv=[]
        q_shifts_all_for_conv_o=[]
        q_shifts_all_for_conv_b=[]
        for id in range(len(latent)):
           
            q_shifts_id = quantize(
                            latent[id] * self.encoder_gains_sf,
                            self.quantizer_noise_type if self.training else "none",
                            self.quantizer_type if self.training else "hardround",
                            self.soft_round_temperature,
                            self.noise_parameter,)

            q_shifts_all_for_conv_o.append(q_shifts_id*self.mask_sf[len(latent)-id-1])
            q_shifts_all_for_conv_b.append(q_shifts_id*(~self.mask_sf[len(latent)-id-1].bool()))
            q_shifts_all_for_conv.append(q_shifts_id)
        
        q_upsample_conv=(self.upsampling_2d(q_shifts_all_for_conv, self.mask_sf))
      



        weight_shift_all,weight_shift_all_b=self.conv_mod(coords, q_upsample_conv)


        return q_shifts_all_for_conv,weight_shift_all,weight_shift_all_b
    

  
    def get_param(self):
       
      
        param = OrderedDict()
       
        param.update({f"conv_mod.{k}": v for k, v in self.conv_mod.get_param().items()})
        param.update({f"arm.{k}": v for k, v in self.arm.get_param().items()})
        param.update({f"upsampling_2d.{k}": v for k, v in self.upsampling_2d.get_param().items()})
        param.update({f"modulation_sf.{i}": v for i, v in enumerate(self.modulation_sf)})
        return param

        
        
    def set_param(self, param):
       
       
        conv_mod_param = {k[len("conv_mod.") :]: v for k, v in param.items() if k.startswith("conv_mod.")}
        arm_param = {k[len("arm.") :]: v for k, v in param.items() if k.startswith("arm.")}
        upsampling_param = {k[len("upsampling_2d.") :]: v for k, v in param.items() if k.startswith("upsampling_2d.")}

        self.conv_mod.set_param(conv_mod_param)
        self.arm.set_param(arm_param)
        self.upsampling_2d.set_param(upsampling_param)
        modulation_sf_param = {int(k.split(".")[1]): v for k, v in param.items() if k.startswith("modulation_sf.")}
        for i, v in modulation_sf_param.items():
            self.modulation_sf[i].data.copy_(v.data)  
    
    def estimate_rate(self, decoder_side_latent,arm_model):
        flat_context = torch.cat(
            [
                _get_neighbor(spatial_latent_i, self.mask_size, self.non_zero_pixel_ctx_index)
                for i,spatial_latent_i in enumerate(decoder_side_latent)
            ],
            dim=0,
        )
        
        flat_latent = torch.cat(
            [spatial_latent_i.view(-1) for i,spatial_latent_i in enumerate(decoder_side_latent)],
            dim=0
        )
       
        flat_context_in=flat_context.unsqueeze(0).transpose(1, 2)
        
        flat_mu, flat_scale, flat_log_scale__ = arm_model(flat_context_in)
        proba = torch.clamp_min(
            _laplace_cdf(flat_latent + 0.5, flat_mu, flat_scale)
            - _laplace_cdf(flat_latent - 0.5, flat_mu, flat_scale),
            min=2**-16,  
        )
        flat_rate = -torch.log2(proba)
        return flat_rate
    def get_network_rate(self):
        
        rate_per_module: DescriptorCoolChic = {
            module_name: {"weight": 0.0, "bias": 0.0}
            for module_name in self.modules_to_send
        }

        for module_name in self.modules_to_send:
            cur_module = getattr(self, module_name)
            rate_per_module[module_name] = measure_expgolomb_rate(
                cur_module,
                self.nn_q_step.get(module_name),
                self.nn_expgol_cnt.get(module_name),
            )
        return rate_per_module


    def compute_rate(self):
        all_score_list=[]
        for layer_id, layer in enumerate(self.net):
            all_score_list.append(layer.scores.view(-1))
        all_score=torch.cat(all_score_list,dim=0)
        num_top_20_percent = int(len(all_score) * (1-self.sparsity))
        topk_values, _ = torch.topk(all_score, num_top_20_percent)
        threshold = topk_values.min().item()
        out_num=[]
        for k in range(len(all_score_list)):
            out_num.append(torch.sum(all_score_list[k]>=threshold).item())
        return out_num

    
    def forward(self, coords, in_mask=None):
        saved_mask=[]
        if self.pe_flag==1:
            input_ = self.pe(coords)
        else:
            input_=coords
            
        q_shifts_all_viewed,weighted_q_shift_all,weight_shift_all_b=self.quantize_all_latent(self.modulation_sf,coords)
        
        input_ = weighted_q_shift_all
        input_1 = weight_shift_all_b
        mask_b = ~self.mask_sf[0]
        flat_rate= self.estimate_rate(q_shifts_all_viewed,self.arm)
        batch_size = input_.shape[0]
        input_ = input_.view(batch_size,3, -1)[:,:,self.mask_sf[0].flatten().bool()]
        input_1 = input_1.view(batch_size,3, -1)[:,:,mask_b.flatten().bool()]
        total_length = self.h*self.w
        
        concatenated_input = torch.zeros(batch_size, 3, total_length,device=input_.device)
        concatenated_input[:, :, self.mask_sf[0].flatten().bool()] = input_
        concatenated_input[:, :, mask_b.flatten().bool()] = input_1
        return concatenated_input.permute(0,2,1),flat_rate,saved_mask