import os
from asyncio import base_tasks
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
import argparse
import random
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch import Tensor, index_select, nn

from utils.quantizer import quantize
from utils.arm import (
    Arm,
    _get_neighbor,
    _get_non_zero_pixel_ctx_index,
    _laplace_cdf,
)
from utils.upsampling import Upsampling
from utils.eval_model import eval_model,compute_model_rate

from utils.quantizemodel import quantize_model
from enc.utils.misc import (
    MAX_ARM_MASK_SIZE,
    POSSIBLE_DEVICE,
    DescriptorCoolChic,
    DescriptorNN,
    measure_expgolomb_rate,
)
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, TypedDict
from itertools import islice
class PosEncodingNeRF(nn.Module):
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


class MultiRegionBlock(nn.Module):

    def __init__(self, K, in_channels, global_hid_channels, local_hid_channels,
                 out_channels):
        super().__init__()
        self.K = K

        self.region_nets = nn.ModuleList([
            nn.Sequential(
                SynthesisLayer(2, local_hid_channels, 1, nn.GELU()),
                SynthesisResidualLayer(local_hid_channels, local_hid_channels, 1, nn.GELU()),
                SynthesisResidualLayer(local_hid_channels, local_hid_channels, 1, nn.GELU()),
                SynthesisResidualLayer(local_hid_channels, 3, 1),
            ) for _ in range(K)
        ])
        self.agg_func = nn.ModuleList([
            SynthesisLayer(global_hid_channels + 6,  3, 1, nn.GELU()),
            SynthesisLayer(global_hid_channels + 9,  3, 1, nn.GELU()),
            SynthesisLayer(global_hid_channels + 12, 3, 1, nn.GELU()),
        ])
        self.full_net = nn.Sequential(
            SynthesisLayer(in_channels, global_hid_channels, 1, nn.GELU()),
            SynthesisLayer(global_hid_channels, 3, 1, nn.GELU()),
            SynthesisResidualLayer(3, 3, 3, nn.GELU()),
            SynthesisResidualLayer(3, 3, 3, nn.GELU()),
        )

    def get_param(self) -> OrderedDict[str, Tensor]:
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        self.load_state_dict(param)

    def forward(self, coordinate, combined_latent, masks_full_res):
        device = combined_latent.device
        masks_full_res = masks_full_res.to(device).bool()

        all_outputs = []
        x = combined_latent
        for layer in self.full_net:
            x = layer(x)
            all_outputs.append(x)

        out_full = [
            torch.cat(all_outputs[:2], dim=1),
            torch.cat(all_outputs[:3], dim=1),
            torch.cat(all_outputs,     dim=1),
        ]

        agg_layers = list(self.agg_func)
        B = combined_latent.shape[0]
        H, W = masks_full_res.shape[-2], masks_full_res.shape[-1]
        output = torch.zeros(B, 3, H, W, device=device, dtype=combined_latent.dtype)

        for k in range(self.K):
            mask_k = masks_full_res[k]
            mask_k_b1hw = mask_k.unsqueeze(0).unsqueeze(0)

            local_input = coordinate * mask_k_b1hw
            net_k = list(self.region_nets[k].children())

            for i in range(3):
                local_input = net_k[i](local_input)
                full_k = torch.where(
                    mask_k_b1hw.expand_as(out_full[i]),
                    out_full[i],
                    torch.zeros_like(out_full[i]),
                )
                local_input = agg_layers[i](torch.cat([local_input, full_k], dim=-3))

            local_input = net_k[3](local_input)
            output = output + local_input * mask_k_b1hw

        return output


class Masked_INR(nn.Module):
    def __init__(self, args, region_map, sparsity, in_features, out_features,
                 hidden_features, hidden_layers, num_regions=None):
        super().__init__()
        self.sparsity = sparsity
        self.net = []

        if region_map.dim() == 4:
            rmap = region_map.squeeze(0).squeeze(0).long()
        elif region_map.dim() == 2:
            rmap = region_map.long()
        else:
            raise ValueError(f"region_map must be 2-D or 4-D, got {tuple(region_map.shape)}")

        self.h = rmap.shape[-2]
        self.w = rmap.shape[-1]
        self.K = int(num_regions) if num_regions is not None else int(rmap.max().item()) + 1
        self.region_map = rmap
        self.pe_flag=0
        if self.pe_flag==1:
            self.pe=PosEncodingNeRF(2,(self.h,self.w))
            input_dim=30
        else:
            input_dim=2

       
        ups_preconcat_k = int(getattr(args, 'upsampling_preconcat_kernel_size', 7))
        n_kern = max(1, int(args.mod_base) - 1)
        self.upsampling_2d = Upsampling(
            ups_k_size=args.local_upsampling_kernel_size,
            ups_preconcat_k_size_or_static=ups_preconcat_k,
            n_ups_kernel_or_highest=n_kern,
            n_ups_preconcat_kernel=n_kern,
        )
        self.dim_arm=args.dim_arm_mod
        self.n_hidden_layers_arm=2
        arm_context_num = args.context_arm
        self.arm = Arm(arm_context_num, args.dim_arm_mod, self.n_hidden_layers_arm)

        self.quantizer_type="softround"
        self.quantizer_noise_type="gaussian"
        self.soft_round_temperature=0.35
        self.noise_parameter=0.22
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
        self.encoder_gains_sf = getattr(args, 'encoder_gain', 16)
        print('Quantizer parameter: encoding gain ',self.encoder_gains_sf)

        self.all_pix_num=self.h*self.w//args.scale//args.scale
        print('total pixel:',self.all_pix_num)
        
        self.register_buffer(
            "non_zero_pixel_ctx_index",
            _get_non_zero_pixel_ctx_index(args.context_arm),
            persistent=False,
        )
       
        self.latent_factor = args.latent_factor

        self.conv_mod = MultiRegionBlock(
            K=self.K,
            in_channels=self.modulation_base_number,
            global_hid_channels=args.sythesis_features,
            local_hid_channels=3,
            out_channels=hidden_layers + 1,
        )

        self.modules_to_send = ['arm', 'conv_mod', 'upsampling_2d']
        self.nn_q_step: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }
        self.nn_expgol_cnt: Dict[str, DescriptorNN] = {
            k: {"weight": None, "bias": None} for k in self.modules_to_send
        }

        self.modulation_sf = nn.ParameterList()

        self.region_mask_sf = []
        one_hot = F.one_hot(rmap.to('cuda'), num_classes=self.K).permute(2, 0, 1).float()
        cur_one_hot = one_hot.unsqueeze(0)

        for layer_idx in range(self.modulation_base_number):
            mod_shape = self.fact_shape[layer_idx]
            shits = nn.Parameter(
                torch.zeros(args.batch_size, 1, mod_shape[0], mod_shape[1])
            ).cuda()
            self.modulation_sf.append(shits)

            if layer_idx > 0:
                cur_one_hot = F.avg_pool2d(cur_one_hot, kernel_size=2, stride=2)
            argmax = cur_one_hot.squeeze(0).argmax(dim=0)
            masks_K = F.one_hot(argmax, num_classes=self.K).permute(2, 0, 1).bool()
            self.region_mask_sf.append(masks_K.cuda())
            print(f'Get Mod with shape {tuple(shits.shape)} at layer {layer_idx+1};'
                  f' region mask {tuple(masks_K.shape)}')

    def quantize_all_latent(self, latent, coords):
        q_main = []
        for id in range(len(latent)):
            q_main.append(quantize(
                latent[id] * self.encoder_gains_sf,
                self.quantizer_noise_type if self.training else "none",
                self.quantizer_type if self.training else "hardround",
                self.soft_round_temperature,
                self.noise_parameter,
            ))

        q_upsample_conv = self.upsampling_2d(q_main)
        full_res_masks = self.region_mask_sf[0]
        recon_image = self.conv_mod(coords, q_upsample_conv, full_res_masks)
        return q_main, recon_image


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

    def estimate_rate(self, decoder_side_latent, arm_model):
        flat_context_list = []
        flat_latent_list = []

        for k, spatial_latent in enumerate(decoder_side_latent):
            s_k = _get_neighbor(
                spatial_latent, self.mask_size, self.non_zero_pixel_ctx_index
            )

            flat_context_list.append(s_k)
            flat_latent_list.append(spatial_latent.view(-1))

        flat_context = torch.cat(flat_context_list, dim=0)
        flat_latent = torch.cat(flat_latent_list, dim=0)
        arm_input = flat_context

        flat_mu, flat_scale, flat_log_scale__ = arm_model(arm_input)
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

    
    def _bucket_rate_per_region(self, flat_rate, decoder_side_latent):
        flat_rate = flat_rate.view(-1)
        K = self.K
        device = flat_rate.device
        rate_per_region = torch.zeros(K, device=device, dtype=flat_rate.dtype)
        pix_per_region  = torch.zeros(K, device=device, dtype=flat_rate.dtype)

        L = self.modulation_base_number

        offset = 0
        for s, latent_i in enumerate(decoder_side_latent):
            n_pix = latent_i.numel()
            rate_i = flat_rate[offset:offset + n_pix]
            offset += n_pix

            mask_i = self.region_mask_sf[L - 1 - s]
            for k in range(K):
                m = mask_i[k].flatten()
                rate_per_region[k] = rate_per_region[k] + rate_i[m].sum()
                pix_per_region[k]  = pix_per_region[k]  + m.sum().to(pix_per_region.dtype)
        return rate_per_region, pix_per_region

    def forward(self, coords, in_mask=None):
        if self.pe_flag == 1:
            _ = self.pe(coords)

        q_main, recon_image = self.quantize_all_latent(self.modulation_sf, coords)
        flat_rate = self.estimate_rate(q_main, self.arm)
        rate_per_region, pix_per_region = self._bucket_rate_per_region(
            flat_rate, q_main
        )

        B = recon_image.shape[0]
        pixels_flat = recon_image.permute(0, 2, 3, 1).reshape(B, -1, 3)
        return pixels_flat, rate_per_region, pix_per_region


