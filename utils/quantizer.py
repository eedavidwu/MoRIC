import math
from typing import Literal, Optional
import typing
import torch
from torch import Tensor


@torch.jit.script
def softround(x: Tensor, t: float) -> Tensor:
   
    floor_x = torch.floor(x)
    delta = x - floor_x - 0.5
    return floor_x + 0.5 * torch.tanh(delta / t) / math.tanh(1 / (2 * t)) + 0.5


@torch.jit.script
def generate_kumaraswamy_noise(
    uniform_noise: Tensor, kumaraswamy_param: float
) -> Tensor:
    
   
    a = kumaraswamy_param
    b = (2**a * (a - 1) + 1) / a

 
    kumaraswamy_noise = (1 - (1 - uniform_noise) ** (1 / b)) ** (1 / a) - 0.5

    return kumaraswamy_noise


POSSIBLE_QUANTIZATION_NOISE_TYPE = Literal["kumaraswamy", "gaussian", "none"]
POSSIBLE_QUANTIZER_TYPE = Literal["softround_alone", "softround", "hardround", "ste", "none"]


def quantize(
    x: Tensor,
    quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy",
    quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround",
    soft_round_temperature: Optional[float] = 0.3,
    noise_parameter: Optional[float] = 1.0,
) -> Tensor:
   
    
    assert quantizer_noise_type in typing.get_args(POSSIBLE_QUANTIZATION_NOISE_TYPE), (
        f"quantizer_noise_type must be in {POSSIBLE_QUANTIZATION_NOISE_TYPE}"
        f" found {quantizer_noise_type}"
    )

    assert quantizer_type in typing.get_args(POSSIBLE_QUANTIZER_TYPE), (
        f"quantizer_type must be in {POSSIBLE_QUANTIZER_TYPE}" f"found {quantizer_type}"
    )

    
    if quantizer_type in ["softround_alone", "hardround", "ste", "none"]:
        if quantizer_noise_type != "none":
            s = (
                f"Using quantizer type {quantizer_type} does not require"
                "to have any random noise.\nSwitching the "
                f"quantizer_noise_type from {quantizer_noise_type} to none."
            )
            print(s)
        quantizer_noise_type = "none"
    else:
        assert quantizer_noise_type != "none", (
            "Using quantizer_noise_type = 'none' is only possible with "
            "quantizer_type = 'softround_alone', 'ste' or 'hardround'.\nTrying"
            f" to use {quantizer_type} which do require some kind of random"
            "noise such as 'gaussian' or 'kumaraswamy'."
        )

   
    match quantizer_noise_type:
        case "none":
            pass
        case "gaussian":
            noise = torch.randn_like(x, requires_grad=False) * noise_parameter
        case "kumaraswamy":
            noise = generate_kumaraswamy_noise(
                torch.rand_like(x, requires_grad=False), noise_parameter
            )
        case _:
            print(f"Unknown quantizer_noise_type {quantizer_noise_type}")

    match quantizer_type:
        case "none":
            return x + noise
        case "softround_alone":
            return softround(x, soft_round_temperature)
        case "softround":
            return softround(
                softround(x, soft_round_temperature) + noise,
                soft_round_temperature,
            )
        case "ste":
            
            y = softround(x, soft_round_temperature)
            with torch.no_grad():
                y = y - softround(x, soft_round_temperature) + torch.round(x)
            return y
        case "hardround":
            return torch.round(x)
        case _:
            print(f"Unknown quantizer_type {quantizer_type}")
