
from typing import List, OrderedDict
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


class UpsamplingConvTranspose2d(nn.Module):
   

    kernel_bilinear = torch.tensor(
        [
            [0.0625, 0.1875, 0.1875, 0.0625],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.0625, 0.1875, 0.1875, 0.0625],
        ]
    )

    kernel_bicubic = torch.tensor(
        [
            [ 0.0012359619 , 0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 , 0.0037078857 , 0.0012359619],
            [ 0.0037078857 , 0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 , 0.0111236572 , 0.0037078857],
            [-0.0092010498 ,-0.0276031494 , 0.0684967041 , 0.2300262451 , 0.2300262451 , 0.0684967041 ,-0.0276031494 ,-0.0092010498],
            [-0.0308990479 ,-0.0926971436 , 0.2300262451 , 0.7724761963 , 0.7724761963 , 0.2300262451 ,-0.0926971436 ,-0.0308990479],
            [-0.0308990479 ,-0.0926971436 , 0.2300262451 , 0.7724761963 , 0.7724761963 , 0.2300262451 ,-0.0926971436 ,-0.0308990479],
            [-0.0092010498 ,-0.0276031494 , 0.0684967041 , 0.2300262451 , 0.2300262451 , 0.0684967041 ,-0.0276031494 ,-0.0092010498],
            [ 0.0037078857 , 0.0111236572 ,-0.0276031494 ,-0.0926971436 ,-0.0926971436 ,-0.0276031494 , 0.0111236572 , 0.0037078857],
            [ 0.0012359619 , 0.0037078857 ,-0.0092010498 ,-0.0308990479 ,-0.0308990479 ,-0.0092010498 , 0.0037078857 , 0.0012359619],
        ]
    )


    def __init__(
        self,
        upsampling_kernel_size: int,
        static_upsampling_kernel: bool
    ):
       
        super().__init__()

        assert upsampling_kernel_size >= 4, (
            f"Upsampling kernel size should be >= 4." f"Found {upsampling_kernel_size}"
        )

        assert upsampling_kernel_size % 2 == 0, (
            f"Upsampling kernel size should be even." f"Found {upsampling_kernel_size}"
        )

        self.upsampling_kernel_size = upsampling_kernel_size
        self.static_upsampling_kernel = static_upsampling_kernel

       
        self.weight = nn.Parameter(
            torch.empty(1, 1, upsampling_kernel_size, upsampling_kernel_size),
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.empty((1)), requires_grad=True)
        self.initialize_parameters()
        
        if self.static_upsampling_kernel:
            
            self.register_buffer("static_kernel", self.weight.data.clone(), persistent=False)
        else:
            self.static_kernel = None

    def initialize_parameters(self) -> None:
       
        self.bias = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)

       
        K = self.upsampling_kernel_size
        self.upsampling_padding = (K // 2, K // 2, K // 2, K // 2)
        self.upsampling_crop = (3 * K - 2) // 2

        if K < 8:
            kernel_init = UpsamplingConvTranspose2d.kernel_bilinear
        else:
            kernel_init = UpsamplingConvTranspose2d.kernel_bicubic

       
        tmpad = (K - kernel_init.size()[0]) // 2
        upsampling_kernel = F.pad(
            kernel_init.clone().detach(),
            (tmpad, tmpad, tmpad, tmpad),
            mode="constant",
            value=0.0,
        )

       
        upsampling_kernel = rearrange(upsampling_kernel, "k_h k_w -> 1 1 k_h k_w")
        self.weight = nn.Parameter(upsampling_kernel, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
       
        upsampling_weight = (
            self.static_kernel if self.static_upsampling_kernel else self.weight
        )

        x_pad = F.pad(x, self.upsampling_padding, mode="replicate")
        y_conv = F.conv_transpose2d(x_pad, upsampling_weight, stride=2, output_padding=1)

       
        H, W = y_conv.size()[-2:]
        results = y_conv[
            :,
            :,
            self.upsampling_crop : H - self.upsampling_crop,
            self.upsampling_crop : W - self.upsampling_crop,
        ]

        return results


class Upsampling(nn.Module):
   


    def __init__(self, upsampling_kernel_size: int, static_upsampling_kernel: bool,highest_flag=1):
       
        super().__init__()

        self.highest_flag=highest_flag

        self.conv_transpose2d = UpsamplingConvTranspose2d(
            upsampling_kernel_size, static_upsampling_kernel
        )

    def forward(self, decoder_side_latent: List[Tensor], masks: List[Tensor]) -> Tensor:
      
        latent_reversed = (decoder_side_latent)

        upsampled_latent = latent_reversed[0] 
        masks = None
        if masks == None:
           
            for target_tensor in latent_reversed[1:]:
                
                x = rearrange(upsampled_latent, "b c h w -> (b c) 1 h w")
               

                x = self.conv_transpose2d(x)
                x = rearrange(x, "(b c) 1 h w -> b c h w", b=upsampled_latent.shape[0])

               
                x = x[:, :, : target_tensor.shape[-2], : target_tensor.shape[-1]]
                
                upsampled_latent = torch.cat((target_tensor, x), dim=1)
        else:
            upsampled_mask = masks[0]
            for target_tensor,target_mask in zip(latent_reversed[1:], masks[1:]):
                
                x = rearrange(upsampled_latent, "b c h w -> (b c) 1 h w")
               

                x = self.conv_transpose2d(x)
                x = rearrange(x, "(b c) 1 h w -> b c h w", b=upsampled_latent.shape[0])

              
                x = x[:, :, : target_tensor.shape[-2], : target_tensor.shape[-1]]
               
                upsampled_latent = torch.cat((target_tensor, x), dim=1)
        if self.highest_flag==0:
            x = rearrange(upsampled_latent, "b c h w -> (b c) 1 h w")
            x = self.conv_transpose2d(x)
            upsampled_latent = rearrange(x, "(b c) 1 h w -> b c h w", b=upsampled_latent.shape[0])
           
        return upsampled_latent

    def get_param(self) -> OrderedDict[str, Tensor]:
        
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]):
        
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
       
        self.conv_transpose2d.initialize_parameters()
