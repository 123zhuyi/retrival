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


"""Rsqrt kernel test result comparator.

Compares kernel output against golden data. Does NOT depend on tensorflow.
bfloat16 is stored as raw uint16 (upper-16 bits of float32) and converted back
to float32 for tolerance-based comparison.
"""

import sys
import numpy as np
import glob
import os


CURR_DIR = os.path.dirname(os.path.realpath(__file__))

FLOAT_TYPES = {"float32", "float16", "bfloat16"}
INT_TYPES = {"int8", "int16", "int32", "uint8"}
BOOL_TYPES = {"bool"}
SUPPORTED = FLOAT_TYPES | INT_TYPES | BOOL_TYPES

READER_MAP = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": np.uint16,  # stored as raw bfloat16 (uint16)
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def bfloat16_raw_to_float32(arr_u16):
    """Convert bfloat16 (uint16 from upper-16 bits) back to float32."""
    return (arr_u16.astype(np.uint32) << 16).view(np.float32)


def rsqrt_compare(golden_file_lists, output_file_lists, d_type):
    if d_type not in SUPPORTED:
        raise ValueError(f"Unsupported dtype: {d_type}")
    read_as = READER_MAP[d_type]

    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, read_as)
        tmp_gold = np.fromfile(gold, read_as)

        if d_type in FLOAT_TYPES:
            # bfloat16 is stored as uint16 — convert back to float32 for comparison
            if d_type == "bfloat16":
                tmp_out = bfloat16_raw_to_float32(tmp_out)
                tmp_gold = bfloat16_raw_to_float32(tmp_gold)
                diff_res = np.isclose(
                    tmp_out, tmp_gold, rtol=4e-3, atol=4e-3, equal_nan=True
                )
            elif d_type == "float32":
                diff_res = np.isclose(
                    tmp_out, tmp_gold, rtol=1e-4, atol=1e-4, equal_nan=True
                )
            else:  # float16
                diff_res = np.isclose(
                    tmp_out, tmp_gold, rtol=1e-3, atol=1e-3, equal_nan=True
                )
        else:
            diff_res = tmp_out == tmp_gold

        diff_idx = np.where(~diff_res)[0]
        if len(diff_idx) == 0:
            print("PASSED!")
        else:
            print("FAILED!")
            for idx in diff_idx[:5]:
                print(f"index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False
    return data_same


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_data.py <dtype>")
        print("Example: python compare_data.py float32")
        exit(1)
    dtype_arg = sys.argv[1]
    prefix = CURR_DIR + "/" + dtype_arg
    glob_gold = sorted(glob.glob(prefix + "*golden*.bin"))
    glob_out = sorted(glob.glob(prefix + "*output*.bin"))
    ok = rsqrt_compare(glob_gold, glob_out, dtype_arg)
    print("compare result:", ok)
    exit(0 if ok else 1)
