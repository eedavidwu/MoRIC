

import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union

import psutil
import torch
from torch import Tensor, nn
from collections import OrderedDict


POSSIBLE_DEVICE = Literal["cpu", "cuda:0"]


def get_best_device() -> POSSIBLE_DEVICE:
    
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    return device


class TrainingExitCode(Enum):
    END = 0
    REQUEUE = 42


def is_job_over(start_time: float, max_duration_job_min: int = 45) -> bool:
    
    if max_duration_job_min < 0:
        return False

    return (time.time() - start_time) / 60 >= max_duration_job_min


@dataclass
class DescriptorNN:
    
    weight: Optional[Union[int, float, str]] = None
    bias: Optional[Union[int, float, str]] = None


@dataclass
class DescriptorCoolChic:
   

    arm: Optional[DescriptorNN] = None
    upsampling: Optional[DescriptorNN] = None
    synthesis: Optional[DescriptorNN] = None


MAX_ARM_MASK_SIZE = 9

POSSIBLE_Q_STEP_SHIFT = {
    "arm": {
        "weight": torch.linspace(-8, 0, 9, device="cpu"),
        "bias": torch.linspace(-16, 0, 17, device="cpu"),
    },
    "conv_mod": {
        "weight": torch.linspace(-12, 0, 13, device="cpu"),
        "bias": torch.linspace(-24, 0, 25, device="cpu"),
    },
    "net": {
        "weight": torch.linspace(-12, 0, 13, device="cpu"),
        "bias": torch.linspace(-24, 0, 25, device="cpu"),
    },
    "upsampling_2d": {
        "weight": torch.linspace(-12, 0, 13, device="cpu"),
        "bias": torch.linspace(-24, 0, 25, device="cpu"),
    },
    "upsampling_2d_b": {
        "weight": torch.linspace(-12, 0, 13, device="cpu"),
        "bias": torch.linspace(-24, 0, 25, device="cpu"),
    }
}
POSSIBLE_Q_STEP = {
    "arm": {
        "weight": 2.0 ** POSSIBLE_Q_STEP_SHIFT["arm"]["weight"],
        "bias": 2.0 ** POSSIBLE_Q_STEP_SHIFT["arm"]["bias"],
    },
    #POSSIBLE_Q_STEP['arm']['weight']
    "conv_mod": {
        "weight": 2.0 ** POSSIBLE_Q_STEP_SHIFT["conv_mod"]["weight"],
        "bias": 2.0 ** POSSIBLE_Q_STEP_SHIFT["conv_mod"]["bias"],
    },
    "net": {
        "weight": 2.0 ** POSSIBLE_Q_STEP_SHIFT["net"]["weight"],
        "bias": 2.0 ** POSSIBLE_Q_STEP_SHIFT["net"]["bias"],
    },
    "upsampling_2d": {
        "weight": 2.0 ** POSSIBLE_Q_STEP_SHIFT["upsampling_2d"]["weight"],
        "bias": 2.0 ** POSSIBLE_Q_STEP_SHIFT["upsampling_2d"]["bias"],
    },
    "upsampling_2d_b": {
        "weight": 2.0 ** POSSIBLE_Q_STEP_SHIFT["upsampling_2d_b"]["weight"],
        "bias": 2.0 ** POSSIBLE_Q_STEP_SHIFT["upsampling_2d_b"]["bias"],
    },
}

POSSIBLE_EXP_GOL_COUNT = {
    "conv_mod": {
        "weight": torch.linspace(0, 12, 13, device="cpu"),
        "bias": torch.linspace(0, 12, 13, device="cpu"),
    },
    "arm": {
        "weight": torch.linspace(0, 12, 13, device="cpu"),
        "bias": torch.linspace(0, 12, 13, device="cpu"),
    },
    "net": {
        "weight": torch.linspace(0, 12, 13, device="cpu"),
        "bias": torch.linspace(0, 12, 13, device="cpu"),
    },
    "upsampling_2d": {
       "weight": torch.linspace(0, 15, 16, device="cpu"),
       "bias": torch.linspace(0, 15, 16, device="cpu"),
   },
   "upsampling_2d_b": {
       "weight": torch.linspace(0, 15, 16, device="cpu"),
       "bias": torch.linspace(0, 15, 16, device="cpu"),
   },
    
}

FIXED_POINT_FRACTIONAL_BITS = 8  


FIXED_POINT_FRACTIONAL_MULT = 2**FIXED_POINT_FRACTIONAL_BITS

MAX_AC_MAX_VAL = 65535  



def get_q_step_from_parameter_name(
    parameter_name: str, q_step: DescriptorNN
) -> Optional[float]:
   
    if parameter_name.endswith(".weight"):
        current_q_step = q_step.get("weight")
    elif parameter_name.endswith(".bias"):
        current_q_step = q_step.get("bias")
    else:
        print(
            'Parameter name should end with ".weight" or ".bias" '
            f"Found: {parameter_name}"
        )
        current_q_step = None

    return current_q_step


@torch.no_grad()
def measure_expgolomb_rate(
    q_module: nn.Module, q_step: DescriptorNN, expgol_cnt: DescriptorNN
) -> DescriptorNN:
    
    sent_param: DescriptorNN = {"bias": [], "weight": []}
    rate_param: DescriptorNN = {"bias": 0.0, "weight": 0.0}
    
    param = q_module.get_param()
   
    for parameter_name, parameter_value in param.items():
        current_q_step = get_q_step_from_parameter_name(parameter_name, q_step)
       
        if current_q_step is None:
            return rate_param

        
        current_sent_param = (parameter_value / current_q_step).view(-1)

        if parameter_name.endswith(".weight"):
            sent_param["weight"].append(current_sent_param)
        elif parameter_name.endswith(".bias"):
            sent_param["bias"].append(current_sent_param)
        else:
            print(
                'Parameter name should end with ".weight" or ".bias" '
                f"Found: {parameter_name}"
            )
            return rate_param

   
    for k, v in sent_param.items():
        
        if len(v) == 0:
            rate_param[k] = 0.0
            continue

        
        current_expgol_cnt = expgol_cnt[k]
        if current_expgol_cnt is None:
            return rate_param

        v = torch.cat(v)

        rate_param[k] = exp_golomb_nbins(v, count=current_expgol_cnt)

    return rate_param


def exp_golomb_nbins(symbol: Tensor, count: int = 0) -> Tensor:
    
    nbins = (
        2 * torch.floor(torch.log2(symbol.abs() / (2**count) + 1))
        + count
        + 1
        + (symbol != 0)
    )
    res = nbins.sum()
    return res



def mem_info(strinfo: str = "Memory allocated") -> None:
    
    mem_cpu = psutil.Process().memory_info().rss
    mem_cpu_GB = mem_cpu / (1024.0 * 1024.0 * 1024.0)

    mem_gpu = torch.cuda.memory_allocated("cuda:0")
    mem_gpu_GB = mem_gpu / (1024.0 * 1024.0 * 1024.0)

    str_display = (
        f"| {strinfo:30s} cpu:{mem_cpu_GB:7.3f} GB --- gpu:{mem_gpu_GB:7.3f} GB |"
    )
    L = len(str_display)
    print(f'{" "*100}{"-"*L}')
    print(f'{" "*100}{str_display}')
    print(f'{" "*100}{"-"*L}')


proba0MPS = torch.tensor(
    [
        0.9891080263649208,
        0.9746796308915489,
        0.9588652555405722,
        0.9438961210609208,
        0.9289674808078398,
        0.9144650894999015,
        0.8988797291640259,
        0.8849083818638724,
        0.8705505632961241,
        0.8542913027588402,
        0.8408964152537145,
        0.8235910172675731,
        0.8098350556562219,
        0.7937188645720145,
        0.7772227308111015,
        0.7659913470050881,
        0.743033931648849,
        0.7348898852047242,
        0.7178727301215397,
        0.7071067811865476,
        0.6870085695324213,
        0.6729634236899158,
        0.6597996876307916,
        0.6433608266170463,
        0.6299896359774878,
        0.6155722066724582,
        0.6040333034402598,
        0.5832959652701518,
        0.5705795714817147,
        0.5520611562919205,
        0.5412248551068882,
        0.5244946637874729,
        0.5,
        0.4585020216023356,
        0.4528797696244531,
        0.43527528164806206,
        0.42044820762685725,
        0.39685943228600723,
        0.39685943228600723,
        0.37151696582442445,
        0.3535533905932738,
        0.3364817118449579,
        0.32987697769322355,
        0.31499481798874385,
        0.29730177875068026,
        0.2806219957472792,
        0.2726269331663144,
        0.25,
        0.25,
        0.2227349718384631,
        0.2050858697731751,
        0.19842971614300361,
        0.1767766952966369,
        0.16493848884661177,
        0.14865088937534013,
        0.1363134665831572,
        0.125,
        0.10254293488658756,
        0.08838834764831845,
        0.07432544468767006,
        0.0625,
        0.04419417382415922,
        0.03125,
        0.015625,
    ]
)


def bac_state_idx_from_proba_0(p0):
   
    states = torch.argmin((proba0MPS - p0).abs())
   
    state = states.item()

    return state * 2 + 1 
