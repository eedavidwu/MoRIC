

import math
from typing import List, Tuple


def encode_exp_golomb(data: List[int], count: List[int]) -> Tuple[bytearray, int]:

    if len(data) != len(count):
        raise ValueError(
            f"Each data to write must have its exp-golomb count parameter. Found {len(data)} "
            f"data to write and {len(count)} exp-golomb count parameters."
        )

    if min(count) < 0:
        raise ValueError(f"Exp-golomb count should be >= 0. Found min(count) = {min(count)}")

    msg = ""
    for x, count_i in zip(data, count):
        if x <= 0:
            x = -2 * x
        else:
            x = 2 * x - 1

        x = x + 2**count_i - 1
        n_leading_zeros = int(math.log2(x + 1))

        bits = f"{x + 1:b}"
        if n_leading_zeros != 0:
            bits = f"{0:>0{n_leading_zeros}b}{bits}"

        bits = bits[count_i:]
        msg += bits

    n_padding_bits = (8 - len(msg) % 8) % 8
    msg = "0" * n_padding_bits + msg

    msg_as_bytearray = bytearray([int(msg[i : i + 8], base=2) for i in range(0, len(msg), 8)])

    return msg_as_bytearray, n_padding_bits


def decode_exp_golomb(data_bytes: bytearray, n_padding_bits: int, count: List[int]) -> List[int]:

    if isinstance(n_padding_bits, float):
        raise TypeError(f"n_padding_bits must be an int. Found n_padding_bits={n_padding_bits}")

    if min(count) < 0:
        raise ValueError(f"Exp-golomb count should be >= 0. Found min(count) = {min(count)}")

    binary_string = "".join(format(byte, "08b") for byte in data_bytes)
    binary_string = binary_string[n_padding_bits:]

    ptr = 0
    all_values = []
    for count_i in count:
        n_bits_to_read = 1
        while True:
            if binary_string[ptr] == "0":
                n_bits_to_read += 1
                ptr += 1
            else:
                break

        quotient = int(binary_string[ptr : ptr + n_bits_to_read], base=2) - 1
        ptr += n_bits_to_read

        if count_i == 0:
            remainder = 0
        else:
            remainder = int(binary_string[ptr : ptr + count_i], base=2)
            ptr += count_i

        value = 2**count_i * quotient + remainder

        is_positive = value % 2
        if is_positive:
            value = (value + 1) // 2
        else:
            value = -value // 2

        all_values.append(value)

    return all_values
