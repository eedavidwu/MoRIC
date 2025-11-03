
import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, TypedDict, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from enc.utils.misc import POSSIBLE_DEVICE

FRAME_TYPE = Literal["I", "P", "B"]

FRAME_DATA_TYPE = Literal["rgb", "yuv420", "yuv444"]
POSSIBLE_BITDEPTH = Literal[8, 10]


class DictTensorYUV(TypedDict):
   

    y: Tensor
    u: Tensor
    v: Tensor


def yuv_dict_to_device(yuv: DictTensorYUV, device: POSSIBLE_DEVICE) -> DictTensorYUV:
  
    return DictTensorYUV(
        y=yuv.get("y").to(device), u=yuv.get("u").to(device), v=yuv.get("v").to(device)
    )



def convert_444_to_420(yuv444: Tensor) -> DictTensorYUV:
    
    assert yuv444.dim() == 4, f"Number of dimension should be 5, found {yuv444.dim()}"

    b, c, h, w = yuv444.size()
    assert c == 3, f"Number of channel should be 3, found {c}"

   
    y = yuv444[:, 0, :, :].view(b, 1, h, w)

   
    uv = F.interpolate(yuv444[:, 1:3, :, :], scale_factor=(0.5, 0.5), mode="nearest")
    u, v = uv.split(1, dim=1)

    yuv420 = DictTensorYUV(y=y, u=u, v=v)
    return yuv420


def convert_420_to_444(yuv420: DictTensorYUV) -> Tensor:
   
    u = F.interpolate(yuv420.get("u"), scale_factor=(2, 2))
    v = F.interpolate(yuv420.get("v"), scale_factor=(2, 2))
    yuv444 = torch.cat((yuv420.get("y"), u, v), dim=1)
    return yuv444




@dataclass
class FrameData:
    
    bitdepth: POSSIBLE_BITDEPTH
    frame_data_type: FRAME_DATA_TYPE
    data: Union[Tensor, DictTensorYUV]

   
    img_size: Tuple[int, int] = field(init=False)
  
    n_pixels: int = field(init=False)  # Height x Width
  

    def __post_init__(self):
        if self.frame_data_type == "rgb" or self.frame_data_type == "yuv444":
            self.img_size = self.data.size()[-2:]
        elif self.frame_data_type == "yuv420":
            self.img_size = self.data.get("y").size()[-2:]

        self.n_pixels = self.img_size[0] * self.img_size[1]

    def to_device(self, device: POSSIBLE_DEVICE) -> None:
       
        if self.frame_data_type == "rgb" or self.frame_data_type == "yuv444":
            self.data = self.data.to(device)
        elif self.frame_data_type == "yuv420":
            self.data = yuv_dict_to_device(self.data, device)


@dataclass
class Frame:
    
    coding_order: int
    display_order: int
    depth: int = 0
    seq_name: str = ""
    data: Optional[FrameData] = None
    decoded_data: Optional[FrameData] = None
    already_encoded: bool = False
    index_references: List[int] = field(default_factory=lambda: [])

    
    refs_data: List[FrameData] = field(default_factory=lambda: [])

   
    frame_type: FRAME_TYPE = field(init=False)
   

    def __post_init__(self):
        assert len(self.index_references) <= 2, (
            "A frame can not have more than 2 references.\n"
            f"Found {len(self.index_references)} references for frame {self.display_order} "
            f"(display order).\n Exiting!"
        )

        if len(self.index_references) == 2:
            self.frame_type = "B"
        elif len(self.index_references) == 1:
            self.frame_type = "P"
        else:
            self.frame_type = "I"

    def set_frame_data(
        self,
        data: Union[Tensor, DictTensorYUV],
        frame_data_type: FRAME_DATA_TYPE,
        bitdepth: POSSIBLE_BITDEPTH,
    ) -> None:
        
        self.data = FrameData(
            bitdepth=bitdepth, frame_data_type=frame_data_type, data=data
        )

    def set_decoded_data(self, decoded_data: FrameData) -> None:
       
        self.decoded_data = decoded_data

    def set_refs_data(self, refs_data: List[FrameData]) -> None:
        
        assert len(refs_data) == len(self.index_references), (
            f"Trying to load data for "
            f"{len(refs_data)} references but current frame only has {len(self.index_references)} "
            f"references. Frame type is {self.frame_type}."
        )

      
        self.refs_data = refs_data

    def upsample_reference_to_444(self) -> None:
       
        upsampled_refs = []
        for ref in self.refs_data:
            if ref.frame_data_type == "yuv420":
                ref.data = convert_420_to_444(ref.data)
                ref.frame_data_type = "yuv444"

            upsampled_refs.append(ref)

        self.refs_data = upsampled_refs

    def to_device(self, device: POSSIBLE_DEVICE) -> None:
        
        if self.data is not None:
            self.data.to_device(device)

        for index_ref in range(len(self.refs_data)):
            if self.refs_data[index_ref] is not None:
                self.refs_data[index_ref].to_device(device)


@dataclass
class CodingStructure:
    
    intra_period: int
    p_period: int = 0
    seq_name: str = ""

    
    frames: List[Frame] = field(init=False)
   

    def __post_init__(self):
        self.frames = self.compute_gop(self.intra_period, self.p_period)

    def compute_gop(self, intra_period: int, p_period: int) -> List[Frame]:
       
        # I-frame
        frames = [
            Frame(
                coding_order=0,
                display_order=0,
                index_references=[],
                seq_name=self.seq_name,
            )
        ]

        if intra_period == 0 and p_period == 0:
            print("Intra period is 0 and P period is 0: all intra coding!")
            return frames

        assert intra_period % p_period == 0, (
            f"Intra period must be divisible by P period."
            f" Found intra_period = {intra_period} ; p_period = {p_period}."
        )

       
        n_chained_gop = intra_period // p_period

        for index_chained_gop in range(n_chained_gop):
            for index_frame_in_gop in range(1, p_period + 1):
                display_order = index_frame_in_gop + index_chained_gop * p_period

                depth_frame_in_gop = self.get_frame_depth_in_gop(index_frame_in_gop)

                
                delta_time_ref = p_period // 2 ** (depth_frame_in_gop - 1)

               
                if index_frame_in_gop == p_period:
                    refs = [display_order - delta_time_ref]
                
                else:
                    refs = [
                        display_order - delta_time_ref,
                        display_order + delta_time_ref,
                    ]

                if depth_frame_in_gop != 0:
                   
                    coding_order_in_gop = depth_frame_in_gop + sum(
                        [2 ** (x - 2) - 1 for x in range(3, depth_frame_in_gop)]
                    )

                   
                    coding_order_in_gop += (index_frame_in_gop - delta_time_ref) // (
                        2 * delta_time_ref
                    )
                else:
                    coding_order_in_gop = 0
                coding_order = index_chained_gop * p_period + coding_order_in_gop

                frames.append(
                    Frame(
                        coding_order=coding_order,
                        display_order=display_order,
                        index_references=refs,
                        depth=depth_frame_in_gop,
                        seq_name=self.seq_name,
                    )
                )

        return frames

    def pretty_string(self) -> str:
        

        COL_WIDTH = 14

        s = "Coding configuration:\n"
        s += "---------------------\n"

        s += f'{"Frame type":<{COL_WIDTH}}\t{"Coding order":<{COL_WIDTH}}\t{"Display order":<{COL_WIDTH}}\t'
        s += f'{"Ref 1":<{COL_WIDTH}}\t{"Ref 2":<{COL_WIDTH}}\t{"Depth":<{COL_WIDTH}}\t{"Encoded"}\n'

        for idx_coding_order in range(len(self.frames)):
            cur_frame = self.get_frame_from_coding_order(idx_coding_order)

            s += f"{cur_frame.frame_type:<{COL_WIDTH}}\t"
            s += f"{cur_frame.coding_order:<{COL_WIDTH}}\t"
            s += f"{cur_frame.display_order:<{COL_WIDTH}}\t"

            if len(cur_frame.index_references) > 0:
                s += f"{cur_frame.index_references[0]:<{COL_WIDTH}}\t"
            else:
                s += f'{"/":<{COL_WIDTH}}\t'

            if len(cur_frame.index_references) > 1:
                s += f"{cur_frame.index_references[1]:<{COL_WIDTH}}\t"
            else:
                s += f'{"/":<{COL_WIDTH}}\t'

            s += f"{cur_frame.depth:<{COL_WIDTH}}\t"

            s += f"{cur_frame.already_encoded:<{COL_WIDTH}}\t"

            s += "\n"
        return s

    def get_number_of_frames(self) -> int:
        
        return len(self.frames)

    def get_max_depth(self) -> int:
        
        return max([frame.depth for frame in self.frames])

    def get_all_frames_of_depth(self, depth: int) -> List[Frame]:
       
        return [frame for frame in self.frames if frame.depth == depth]

    def get_max_coding_order(self) -> int:
       
        return max([frame.coding_order for frame in self.frames])

    def get_frame_from_coding_order(self, coding_order: int) -> Optional[Frame]:
       
        for frame in self.frames:
            if frame.coding_order == coding_order:
                return frame
        return None

    def get_max_display_order(self) -> int:
       
        return max([frame.display_order for frame in self.frames])

    def get_frame_from_display_order(self, display_order: int) -> Optional[Frame]:
       
        for frame in self.frames:
            if frame.display_order == display_order:
                return frame
        return None

    def set_encoded_flag(self, coding_order: int, flag_value: bool) -> None:
        
        for frame in self.frames:
            if frame.coding_order == coding_order:
                frame.already_encoded = flag_value

    def unload_all_decoded_data(self) -> None:
        
        for idx_display_order in range(self.get_number_of_frames()):
            
            self.frames[idx_display_order].decoded_data = None

    def unload_all_original_frames(self) -> None:
        
        for idx_display_order in range(self.get_number_of_frames()):
           
            self.frames[idx_display_order].data = None

    def unload_all_references_data(self) -> None:
        
        for idx_display_order in range(self.get_number_of_frames()):
            
            self.frames[idx_display_order].refs_data = None

    def get_frame_depth_in_gop(self, idx_frame: int) -> int:
        
        assert idx_frame <= self.p_period, (
            f"idx_frame should be <= to p_period."
            f" P-period is {self.p_period}, Index frame is {idx_frame}."
        )

        assert math.log2(self.p_period) % 1 == 0, (
            f"p_period should be a power of 2." f" P-period is {self.p_period}."
        )

        if idx_frame == 0:
            return 0

        
        depth = int(math.log2(self.p_period) + 1)
        for i in range(int(math.log2(self.p_period)), 0, -1):
            if idx_frame % 2**i == 0:
                depth -= 1

        return int(depth)
