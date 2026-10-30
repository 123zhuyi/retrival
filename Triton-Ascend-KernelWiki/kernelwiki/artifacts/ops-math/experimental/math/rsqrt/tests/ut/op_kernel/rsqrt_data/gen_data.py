#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Rsqrt kernel test data generator.

Generates random input and golden (expected) output for rsqrt kernel UT.
Does NOT depend on tensorflow — bfloat16 is handled via manual bit manipulation.
"""

import sys
import os
import numpy as np


FLOAT_TYPES = {"float32", "float16", "bfloat16"}
INT_TYPES = {"int8", "int16", "int32", "uint8"}
BOOL_TYPES = {"bool"}
SUPPORTED = FLOAT_TYPES | INT_TYPES | BOOL_TYPES

DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": np.uint16,  # stored as raw uint16 (upper-16 bits of float32)
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def float32_to_bfloat16_raw(arr_f32):
    """Convert float32 to bfloat16 (uint16) with round-to-nearest-even, matching AscendC CAST_RINT."""
    u32 = arr_f32.view(np.uint32)
    lsb = (u32 >> 16) & 1  # LSB of the bfloat16 result (bit 16 of float32)
    rounded = u32 + 0x7FFF + lsb  # RNE rounding: add half-ulp + tie-breaking
    return (rounded >> 16).astype(np.uint16)


def bfloat16_raw_to_float32(arr_u16):
    """Convert bfloat16 (uint16 from upper-16 bits) back to float32 (zero-extend lower bits)."""
    return (arr_u16.astype(np.uint32) << 16).view(np.float32)


def bfloat16_rsqrt(arr_f32):
    """Simulate AscendC kernel bfloat16 rsqrt: bf16→f32 (CAST_NONE) → sqrt+div in f32 → f32→bf16 (CAST_RINT)."""
    bf = float32_to_bfloat16_raw(arr_f32)  # truncate input to bf16 precision
    x_f32 = bfloat16_raw_to_float32(bf)  # Cast to float32 (zero-extend lower bits)
    result_f32 = 1.0 / np.sqrt(x_f32)  # full float32 sqrt + div
    return float32_to_bfloat16_raw(result_f32)  # Cast to bfloat16 with RNE rounding


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip("(").strip(")").strip()
    parts = [x.strip() for x in shape_str.split(",") if x.strip() != ""]
    return np.array([int(x) for x in parts])


def gen_data_and_golden(shape_str, d_type="float32"):
    if d_type not in SUPPORTED:
        raise ValueError(f"Unsupported dtype: {d_type}")
    shape = parse_str_to_shape_list(shape_str)
    size = np.prod(shape)

    # 固定随机种子，保证 CI 复现性
    rng = np.random.RandomState(hash(d_type) & 0xFFFFFFFF)

    if d_type in FLOAT_TYPES:
        tmp_input = rng.uniform(0.1, 10.0, size=size).reshape(shape).astype(np.float32)
        if d_type == "float32":
            tmp_golden = 1.0 / np.sqrt(tmp_input)
            np_type = np.float32
        elif d_type == "float16":
            tmp_golden = (1.0 / np.sqrt(tmp_input.astype(np.float16))).astype(
                np.float16
            )
            np_type = np.float16
        else:  # bfloat16
            tmp_golden = bfloat16_rsqrt(tmp_input)
            np_type = np.uint16  # store as raw bfloat16 (uint16)
            tmp_input = float32_to_bfloat16_raw(tmp_input)
            # tmp_golden is already bfloat16 (uint16) from bfloat16_rsqrt
            tmp_input.astype(np_type).tofile(f"{d_type}_input_rsqrt.bin")
            tmp_golden.astype(np_type).tofile(f"{d_type}_golden_rsqrt.bin")
            return

    elif d_type in INT_TYPES:
        tmp_input = rng.randint(1, 100, size=size, dtype=np.int64).reshape(shape)
        tmp_golden_fp = 1.0 / np.sqrt(tmp_input.astype(np.float32))
        np_type = DTYPE_MAP[d_type]
        if d_type == "uint8":
            tmp_golden = np.clip(np.rint(tmp_golden_fp), 0, 255).astype(np.uint8)
        elif d_type == "int8":
            tmp_golden = np.clip(np.rint(tmp_golden_fp), -128, 127).astype(np.int8)
        else:
            tmp_golden = np.rint(tmp_golden_fp).astype(np_type)
        tmp_input = tmp_input.astype(np_type)

    elif d_type in BOOL_TYPES:
        tmp_input = rng.choice([True, False], size=size).reshape(shape)
        tmp_golden = np.ones(shape, dtype=np.bool_)
        np_type = np.bool_

    tmp_input.astype(np_type).tofile(f"{d_type}_input_rsqrt.bin")
    tmp_golden.astype(np_type).tofile(f"{d_type}_golden_rsqrt.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
