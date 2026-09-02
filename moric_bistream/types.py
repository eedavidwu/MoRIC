
from dataclasses import dataclass, fields
from typing import Any, Optional


@dataclass
class DescriptorNN:

    weight: Optional[Any] = None
    bias: Optional[Any] = None

    def get_value(self, weight_or_bias: str) -> Any:
        available_name = [x.name for x in fields(DescriptorNN)]
        if weight_or_bias not in available_name:
            raise ValueError(
                f"Can not get value for weight_or_bias={weight_or_bias}. Available "
                f"names are {available_name}"
            )
        return self.__getattribute__(weight_or_bias)
