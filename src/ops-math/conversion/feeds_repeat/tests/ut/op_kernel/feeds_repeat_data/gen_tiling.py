#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import numpy as np
import sys

params_fp32_int32 = [
    210, #elem_row
    216, #elem_per_loop
    4, #length
    8, #length_aligned
    48, #max_core_num
    12, #core_per_group
    0, #core_moreover
    15, #empty_size
    1, #row_per_core
    0 #row_left
]

params_fp16_int32 = [
    3250, #elem_row
    3264, #elem_per_loop
    1, #length
    8, #length_aligned
    48, #max_core_num
    48, #core_per_group
    0, #core_moreover
    128, #empty_size
    1, #row_per_core
    0 #row_left
]

params_bf16_int32 = [
    21000, #elem_row
    21008, #elem_per_loop
    4, #length
    8, #length_aligned
    48, #max_core_num
    12, #core_per_group
    0, #core_moreover
    15, #empty_size
    1, #row_per_core
    0 #row_left
]

params_fp32_int64 = [
    210, #elem_row
    216, #elem_per_loop
    48, #length
    48, #length_aligned
    48, #max_core_num
    0, #core_per_group
    0, #core_moreover
    100, #empty_size
    1, #row_per_core
    0 #row_left
]

params_fp16_int64 = [
    210, #elem_row
    224, #elem_per_loop
    50, #length
    52, #length_aligned
    48, #max_core_num
    0, #core_per_group
    0, #core_moreover
    100, #empty_size
    1, #row_per_core
    2 #row_left
]

params_bf16_int64 = [
    210, #elem_row
    224, #elem_per_loop
    100, #length
    100, #length_aligned
    48, #max_core_num
    0, #core_per_group
    0, #core_moreover
    101, #empty_size
    2, #row_per_core
    4 #row_left
]

params_info = {
    "case_fp32_int32": params_fp32_int32,
    "case_fp16_int32": params_fp16_int32,
    "case_bf16_int32": params_bf16_int32,
    "case_fp32_int64": params_fp32_int64,
    "case_fp32_int64": params_fp16_int64,
    "case_fp32_int64": params_bf16_int64,
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"

    base_params = np.array(params_list[:], dtype=np.uint32)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)

if __name__ == '__main__':
    main()