
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

class ArmLinear(nn.Module):
  

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool = False,
    ):
        

        super().__init__()

        self.residual = residual

        
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels), requires_grad=True
        )
        self.bias = nn.Parameter(torch.empty((out_channels)), requires_grad=True)
        self.initialize_parameters()
       

    def initialize_parameters(self) -> None:
       
        self.bias = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)
        if self.residual:
            self.weight = nn.Parameter(
                torch.zeros_like(self.weight), requires_grad=True
            )
        else:
            out_channel = self.weight.size()[0]
            self.weight = nn.Parameter(
                torch.randn_like(self.weight) / out_channel**2, requires_grad=True
            )

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass of this layer.

        Args:
            x: Input tensor of shape :math:`[B, C_{in}]`.

        Returns:
            Tensor with shape :math:`[B, C_{out}]`.
        """
        if self.residual:
            return F.linear(x, self.weight, bias=self.bias) + x

      
        else:
            return F.linear(x, self.weight, bias=self.bias)

class ArmIntLinear(nn.Module):
    

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        fpfm: int = 0,
        pure_int: bool = False,
        residual: bool = False,
    ):
       

        super().__init__()

        self.fpfm = fpfm
        self.pure_int = pure_int
        self.residual = residual

      
        if self.pure_int:
            self.weight = nn.Parameter(
                torch.empty((out_channels, in_channels), dtype=torch.int32), requires_grad=False
            )
            self.bias = nn.Parameter(torch.empty((out_channels), dtype=torch.int32), requires_grad=False)
        else:
            self.weight = nn.Parameter(
                torch.empty((out_channels, in_channels), dtype=torch.float), requires_grad=False
            )
            self.bias = nn.Parameter(torch.empty((out_channels), dtype=torch.float), requires_grad=False)
      


    def forward(self, x: Tensor) -> Tensor:
        
        if self.residual:
            xx = F.linear(x, self.weight, bias=self.bias) + x*self.fpfm
        else:
            xx = F.linear(x, self.weight, bias=self.bias)

        
        if self.pure_int:
            xx = xx + torch.sign(xx)*self.fpfm//2
           
            neg_result = -((-xx)//self.fpfm)
            pos_result = xx//self.fpfm
            result = torch.where(xx < 0, neg_result, pos_result)
        else:
            xx = xx + torch.sign(xx)*self.fpfm/2
          
            neg_result = -((-xx)/self.fpfm)
            pos_result = xx/self.fpfm
            result = torch.where(xx < 0, neg_result, pos_result)
            result = result.to(torch.int32).to(torch.float)

        return result


LOG_SCALE_MIN = -5
LOG_SCALE_MAX = 5


class Arm(nn.Module):
    


    def __init__(self, context_num:int, dim_arm: int, n_hidden_layers_arm: int):
        
        super().__init__()

        
        assert 1 <= context_num <= 40, (
            f"ARM context size must be in [1, 40] (capped by the 9x9 mask). "
            f"Found {context_num}."
        )

       
        layers_list = nn.ModuleList()

        
        first_layer_residual = (context_num == dim_arm)
        
        layers_list.append(ArmLinear(context_num, dim_arm, residual=first_layer_residual))
        layers_list.append(nn.ReLU())
        layers_list.append(ArmLinear(dim_arm, dim_arm, residual=True))
        layers_list.append(nn.ReLU())
        
        layers_list.append(ArmLinear(dim_arm, 2, residual=False))

        self.mlp = nn.Sequential(*layers_list)
       
        self.flag_linear_stabiliser = False
        self.dim_arm = dim_arm
        self.n_out_features = 2

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        
        raw_proba_param = self.mlp(x)
        mu = raw_proba_param[:, 0]
        log_scale = raw_proba_param[:, 1]

       
        scale = torch.exp(torch.clamp(log_scale - 4, min=LOG_SCALE_MIN, max=LOG_SCALE_MAX))
        return mu, scale, log_scale

    def get_param(self) -> OrderedDict[str, Tensor]:
       
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        
        self.load_state_dict(param)

    def reinitialize_parameters(self) -> None:
       
        for layer in self.mlp.children():
            if isinstance(layer, ArmLinear):
                layer.initialize_parameters()

class ArmInt(nn.Module):
   

    def __init__(self, dim_arm: int, n_hidden_layers_arm: int, fpfm: int, pure_int: bool):
       
        super().__init__()

        assert dim_arm % 8 == 0, (
            f"ARM context size and hidden layer dimension must be "
            f"a multiple of 8. Found {dim_arm}."
        )

        self.FPFM = fpfm 
        self.pure_int = pure_int 

        
        layers_list = nn.ModuleList()

       
        for i in range(n_hidden_layers_arm):
            layers_list.append(ArmIntLinear(dim_arm, dim_arm, self.FPFM, self.pure_int, residual=True))
            layers_list.append(nn.ReLU())

        
        layers_list.append(ArmIntLinear(dim_arm, 2, self.FPFM, self.pure_int, residual=False))
        self.mlp = nn.Sequential(*layers_list)
        # ======================== Construct the MLP ======================== #

    def set_param_from_float(self, float_param: OrderedDict[str, Tensor]) -> None:
        
        integerised_param = {}
        for k in float_param:
            if "weight" in k:
                float_v = float_param[k]*self.FPFM
            else:
                float_v = float_param[k]*self.FPFM*self.FPFM

            float_v = float_v + torch.sign(float_v)*0.5
            neg_result = -(-float_v).to(torch.int32)
            pos_result = float_v.to(torch.int32)
            int_v = torch.where(float_v < 0, neg_result, pos_result)
            if not self.pure_int:
                int_v = int_v.to(torch.float)
            integerised_param[k] = nn.parameter.Parameter(int_v, requires_grad=False)

        self.load_state_dict(integerised_param, assign=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        
        xint = x.clone().detach()
        xint = xint*self.FPFM
        if self.pure_int:
            xint = xint.to(torch.int32)

        for idx_l, layer in enumerate(self.mlp.children()):
            xint = layer(xint)

        
        raw_proba_param = xint / self.FPFM

        mu = raw_proba_param[:, 0]
        log_scale = raw_proba_param[:, 1]

        
        scale = torch.exp(torch.clamp(log_scale - 4, min=-5, max=5.0))

        return mu, scale, log_scale

    def get_param(self) -> OrderedDict[str, Tensor]:
        
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        
        self.load_state_dict(param)

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


_MAX_ARM_MASK_SIZE = 9


def get_priority_order() -> Tensor:
    
    return torch.tensor(
        [
            38, 35, 30, 25, 23, 31, 36, 37, 39,
            33, 28, 21, 20,  6, 15, 22, 29, 34,
            32, 18, 12, 10,  5,  9, 14, 19, 27,
            24, 13,  8,  2,  1,  3, 11, 17, 26,
            16,  7,  4,  0,  #
        ]
    )
   


def _get_mask_size_ctx(n_spatial_ctx: int = 0) -> int:
    
    return _MAX_ARM_MASK_SIZE


def _get_non_zero_pixel_ctx_index(n_spatial_ctx: int) -> Tensor:
    
    center_pixel_idx = (_get_mask_size_ctx(n_spatial_ctx) ** 2 - 1) // 2  # = 40
    possible_neighbors = torch.arange(center_pixel_idx)
    selected_neighbors = possible_neighbors[
        get_priority_order().argsort(stable=True)
    ][:n_spatial_ctx]
    return selected_neighbors
