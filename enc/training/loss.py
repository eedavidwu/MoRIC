import math
from dataclasses import dataclass, field
from typing import Optional, Union
import torch
from enc.utils.codingstructure import DictTensorYUV
from torch import Tensor

@dataclass(kw_only=True)
class LossFunctionOutput():
   
 
    loss: Optional[float] = None                                     

  
    mse: Optional[float] = None                                        
    rate_nn_bpp: Optional[float] = None                               
    rate_latent_bpp: Optional[float] = None                             


    psnr_db: Optional[float] = field(init=False, default=None)                               
    total_rate_bpp: Optional[float] = field(init=False, default=None)   


    def __post_init__(self):
        if self.mse is not None:
            self.psnr_db = -10.0 * math.log10(self.mse)

        if self.rate_nn_bpp is not None and self.rate_latent_bpp is not None:
            self.total_rate_bpp = self.rate_nn_bpp + self.rate_latent_bpp


def _compute_mse(
    x: Union[Tensor, DictTensorYUV], y: Union[Tensor, DictTensorYUV]
) -> Tensor:
    
    flag_420 = not (isinstance(x, Tensor))

    if not flag_420:
        return ((x - y) ** 2).mean()
    else:
       
        total_pixels_yuv = 0.0

        
        mse = torch.zeros((1), device=x.get("y").device)
        for (_, x_channel), (_, y_channel) in zip(x.items(), y.items()):
            n_pixels_channel = x_channel.numel()
            mse = (
                mse + torch.pow((x_channel - y_channel), 2.0).mean() * n_pixels_channel
            )
            total_pixels_yuv += n_pixels_channel
        mse = mse / total_pixels_yuv
        return mse


def loss_function(
    decoded_image: Union[Tensor, DictTensorYUV],
    rate_latent_bit: Tensor,
    target_image: Union[Tensor, DictTensorYUV],
    lmbda: float = 1e-3,
    rate_mlp_bit: float = 0.0,
    compute_logs: bool = False,
) -> LossFunctionOutput:
    

    mse = _compute_mse(decoded_image, target_image)

    n_pixels=decoded_image.shape[1]
   
    rate_MLP=rate_mlp_bit/n_pixels
    rate_bpp = (rate_latent_bit.sum() + rate_mlp_bit) / n_pixels

    loss = mse + lmbda * rate_bpp

    output = LossFunctionOutput(
        loss=loss,
        mse=mse.detach().item() if compute_logs else None,
        rate_nn_bpp=rate_mlp_bit / n_pixels if compute_logs else None,
        rate_latent_bpp=rate_latent_bit.detach().sum().item() / n_pixels
        if compute_logs
        else None,
    )

    return output
