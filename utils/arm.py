


from typing import OrderedDict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, index_select, nn


class ArmConv(nn.Module):
   
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool = False,
    ):
        super().__init__()

        self.residual = residual
        self.conv1_1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
       

    def forward(self, x: Tensor) -> Tensor:
       
        if self.residual:
            return self.conv1_1(x) + x

       
        else:
            return self.conv1_1(x)


class Arm(nn.Module):
   


    def __init__(self, context_num:int, dim_arm: int, n_hidden_layers_arm: int):
       
        super().__init__()

        assert context_num % 8 == 0, (
            f"ARM context size and hidden layer dimension must be "
            f"a multiple of 8. Found {context_num}."
        )

      
        layers_list = nn.ModuleList()

      
        layers_list.append(ArmConv(context_num, dim_arm, residual=True))
        layers_list.append(nn.GELU())
        layers_list.append(ArmConv(dim_arm, dim_arm, residual=True))
        layers_list.append(nn.GELU())
        
        layers_list.append(ArmConv(dim_arm, 2, residual=False))

        self.mlp = nn.Sequential(*layers_list)
       

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        
        raw_proba_param = self.mlp(x)
        mu = raw_proba_param[:, 0]
        log_scale = raw_proba_param[:, 1]

        
        scale = torch.exp(torch.clamp(log_scale - 4, min=-4.6, max=5.0))
        return mu, scale, log_scale

    def get_param(self) -> OrderedDict[str, Tensor]:
        
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
       
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
       
        for layer in self.mlp.children():
            if isinstance(layer, ArmLinear):
                layer.initialize_parameters()



#@torch.jit.script
def _get_neighbor(x: Tensor, mask_size: int, non_zero_pixel_ctx_idx: Tensor) -> Tensor:
    
    pad = int((mask_size - 1) / 2)
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)

   
    x_unfold = (
        x_pad.unfold(2, mask_size, step=1)
        .unfold(3, mask_size, step=1)
        .reshape(-1, mask_size * mask_size)
    )

   
    neighbor = index_select(x_unfold, dim=1, index=non_zero_pixel_ctx_idx)
    return neighbor


@torch.jit.script
def _laplace_cdf(x: Tensor, expectation: Tensor, scale: Tensor) -> Tensor:
  
    shifted_x = x - expectation
    return 0.5 - 0.5 * (shifted_x).sign() * torch.expm1(-(shifted_x).abs() / scale)


def _get_non_zero_pixel_ctx_index(dim_arm: int) -> Tensor:

    if dim_arm == 8:
        return torch.tensor(
            [            13,
                         22,
                     30, 31, 32,
             37, 38, 39, #
            ]
        )

    elif dim_arm == 16:
        return torch.tensor(
            [
                            13, 14,
                    20, 21, 22, 23, 24,
                28, 29, 30, 31, 32, 33,
                37, 38, 39, #
            ]
        )

    elif dim_arm == 24:
        return torch.tensor(
            [
                                4 ,
                        11, 12, 13, 14, 15,
                    19, 20, 21, 22, 23, 24, 25,
                    28, 29, 30, 31, 32, 33, 34,
                36, 37, 38, 39, #
            ]
        )

    elif dim_arm == 32:
        return torch.tensor(
            [
                        2 , 3 , 4 , 5 ,
                    10, 11, 12, 13, 14, 15, 16,
                    19, 20, 21, 22, 23, 24, 25, 26,
                27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, #
            ]
        )
