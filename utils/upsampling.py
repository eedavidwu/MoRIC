

from typing import List, Optional, OrderedDict

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from einops import rearrange
from torch import Tensor, nn


class _Parameterization_Symmetric_1d(nn.Module):
    

    def __init__(self, target_k_size: int):
        super().__init__()
        self.target_k_size = target_k_size
        self.param_size = _Parameterization_Symmetric_1d.size_param_from_target(self.target_k_size)

    def forward(self, x: Tensor) -> Tensor:
        x_reversed = torch.fliplr(x.view(1, -1)).view(-1)
       
        kernel = torch.cat([x, x_reversed[self.target_k_size % 2:]])
        return kernel

    @classmethod
    def size_param_from_target(cls, target_k_size: int) -> int:
       
        return (target_k_size + 1) // 2


class UpsamplingSeparableSymmetricConv2d(nn.Module):
   

    def __init__(self, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, f"Upsampling pre-concat kernel size must be odd. Found {kernel_size}."
        self.target_k_size = kernel_size
        self.param_size = _Parameterization_Symmetric_1d.size_param_from_target(self.target_k_size)

        self.weight = nn.Parameter(torch.empty(self.param_size), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(1), requires_grad=True)
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """Init kernel as a Dirac (identity) so output = input at init.
        Bias is zero."""
        if parametrize.is_parametrized(self, "weight"):
            parametrize.remove_parametrizations(self, "weight", leave_parametrized=False)

        # Half-kernel: (0, ..., 0, 1). After symmetric expansion -> (0, ..., 0, 1, 0, ..., 0)
        w = torch.zeros_like(self.weight)
        w[-1] = 1
        self.weight = nn.Parameter(w, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)

        parametrize.register_parametrization(
            self,
            "weight",
            _Parameterization_Symmetric_1d(target_k_size=self.target_k_size),
            unsafe=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        k = self.weight.size()[0]
        weight = self.weight.view(1, -1)
        padding = k // 2

        if x.size()[1] == 0:
            return x

        if self.training:
            # Non-separable for stability: build 2D kernel as outer product
            kernel_2d = torch.kron(weight, weight.T).view((1, 1, k, k))
            return F.conv2d(x, kernel_2d, bias=None, stride=1, padding=padding) + x
        else:
            # Separable: two 1D convs in sequence
            yw = F.conv2d(x, weight.view((1, 1, 1, k)), padding=(0, padding))
            return F.conv2d(yw, weight.view((1, 1, k, 1)), padding=(padding, 0)) + x


class UpsamplingSeparableSymmetricConvTranspose2d(nn.Module):
    """Separable + symmetric 2D transposed conv with an *even* kernel
    performing the x2 spatial upsampling."""

    def __init__(self, kernel_size: int):
        super().__init__()
        assert kernel_size >= 4 and not kernel_size % 2, (
            f"Upsampling TConv kernel size must be even and >=4. Found {kernel_size}."
        )
        self.target_k_size = kernel_size
        self.param_size = _Parameterization_Symmetric_1d.size_param_from_target(self.target_k_size)

        self.weight = nn.Parameter(torch.empty(self.param_size), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(1), requires_grad=True)
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
       
        if parametrize.is_parametrized(self, "weight"):
            parametrize.remove_parametrizations(self, "weight", leave_parametrized=False)

        if self.target_k_size < 8:
            kernel_core = torch.tensor([1.0 / 4.0, 3.0 / 4.0])
        else:
            kernel_core = torch.tensor([0.0351562, 0.1054687, -0.2617187, -0.8789063])

        zero_pad = self.param_size - kernel_core.size()[0]
        w = torch.zeros_like(self.weight)
        w[zero_pad:] = kernel_core
        self.weight = nn.Parameter(w, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)

        parametrize.register_parametrization(
            self,
            "weight",
            _Parameterization_Symmetric_1d(target_k_size=self.target_k_size),
            unsafe=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        k = self.target_k_size
        P0 = k // 2
        C = 2 * P0 - 1 + k // 2  # crop side border  (k=4 -> 5; k=8 -> 11)

        weight = self.weight.view(1, -1)

        if self.training:
            kernel_2d = torch.kron(weight, weight.T).view((1, 1, k, k))
            x_pad = F.pad(x, (P0, P0, P0, P0), mode="replicate")
            yc = F.conv_transpose2d(x_pad, kernel_2d, stride=2)
            H, W = yc.size()[-2:]
            y = yc[:, :, C: H - C, C: W - C]
        else:
            # Horizontal then vertical
            x_pad = F.pad(x, (P0, P0, 0, 0), mode="replicate")
            yc = F.conv_transpose2d(x_pad, weight.view((1, 1, 1, k)), stride=(1, 2))
            W = yc.size()[-1]
            y = yc[:, :, :, C: W - C]
            x_pad = F.pad(y, (0, 0, P0, P0), mode="replicate")
            yc = F.conv_transpose2d(x_pad, weight.view((1, 1, k, 1)), stride=(2, 1))
            H = yc.size()[-2]
            y = yc[:, :, C: H - C, :]
        return y


class Upsampling(nn.Module):
   

    def __init__(
        self,
        ups_k_size: int = 8,
        ups_preconcat_k_size_or_static=None,
        n_ups_kernel_or_highest=None,
        n_ups_preconcat_kernel: Optional[int] = None,
    ):
        super().__init__()

       
        if (
            isinstance(ups_preconcat_k_size_or_static, bool)
            or ups_preconcat_k_size_or_static is None
            or (isinstance(ups_preconcat_k_size_or_static, int) and isinstance(n_ups_kernel_or_highest, int) and n_ups_preconcat_kernel is None and n_ups_kernel_or_highest in (0, 1))
        ):
           
            ups_preconcat_k_size = 7  
            n_ups_kernel = 1          
            n_ups_preconcat_kernel = 1
        else:
            ups_preconcat_k_size = int(ups_preconcat_k_size_or_static)
            n_ups_kernel = int(n_ups_kernel_or_highest) if n_ups_kernel_or_highest is not None else 1
            n_ups_preconcat_kernel = int(n_ups_preconcat_kernel) if n_ups_preconcat_kernel is not None else n_ups_kernel

        self.ups_k_size = ups_k_size
        self.ups_preconcat_k_size = ups_preconcat_k_size
        self.n_ups_kernel = n_ups_kernel
        self.n_ups_preconcat_kernel = n_ups_preconcat_kernel

      
        self.conv_transpose2ds = nn.ModuleList(
            [UpsamplingSeparableSymmetricConvTranspose2d(ups_k_size) for _ in range(n_ups_kernel)]
        )
        self.conv2ds = nn.ModuleList(
            [UpsamplingSeparableSymmetricConv2d(ups_preconcat_k_size) for _ in range(n_ups_preconcat_kernel)]
        )

   
    def forward(self, decoder_side_latent: List[Tensor], masks=None) -> Tensor:
       
        upsampled_latent = decoder_side_latent[0]

        for idx, target_tensor in enumerate(decoder_side_latent[1:]):
            if target_tensor.size()[1] == 0:
                break

          
            x = rearrange(upsampled_latent, "b c h w -> (b c) 1 h w")
            x = self.conv_transpose2ds[idx % self.n_ups_kernel](x)
            x = rearrange(x, "(b c) 1 h w -> b c h w", b=upsampled_latent.shape[0])
            
            x = x[:, :, : target_tensor.shape[-2], : target_tensor.shape[-1]]

           
            high_branch = self.conv2ds[idx % self.n_ups_preconcat_kernel](target_tensor)

           
            upsampled_latent = torch.cat((high_branch, x), dim=1)

        return upsampled_latent

    
    def get_param(self) -> OrderedDict[str, Tensor]:
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]) -> None:
        self.load_state_dict(param, strict=False)

    def reinitialize_parameters(self) -> None:
        for m in self.conv_transpose2ds:
            m.initialize_parameters()
        for m in self.conv2ds:
            m.initialize_parameters()
