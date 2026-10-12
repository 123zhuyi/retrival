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

case0_params = [4, 1, 1, 0, 10, 1, 4, 0, 10, 128, 119, 5, 0] # no group_idx float16
case1_params = [16, 16, 1, 0, 17, 16, 2000, 128, 0, 128, 119, 17, 0] # no group_idx float32
case2_params = [6, 6, 1, 0, 1, 3, 0, 256, 100, 128, 119, 0, 0] # group_idx bfloat16
case3_params = [12, 12, 1, 0, 17, 3, 0, 458, 1968, 128, 119, 0, 0] # group_idx float16
case4_params = [16, 16, 1, 0, 27, 4, 0, 399, 3200, 128, 119, 0, 0] # group_idx float32
case5_params = [40, 40, 4, 3, 75, 16, 0, 1228, 8875, 128, 119, 0, 0] # group_idx float16
case6_params = [6, 6, 1, 0, 1, 3, 0, 256, 100, 128, 119, 0, 0] # group_idx float16
case7_params = [40, 20, 112, 112, 18, 224, 0, 2560, 1968, 128, 115, 0, 0] # group_idx float32

# group idx dtype is int64
case8_params = [6, 6, 1, 0, 1, 3, 0, 256, 100, 128, 119, 0, 0] # group_idx bfloat16
case9_params = [12, 12, 1, 0, 17, 3, 0, 458, 1968, 128, 119, 0, 0] # group_idx float16
case10_params = [16, 16, 1, 0, 27, 4, 0, 399, 3200, 128, 119, 0, 0] # group_idx float32
case11_params = [40, 40, 4, 3, 75, 16, 0, 1228, 8875, 128, 119, 0, 0] # group_idx float16
case12_params = [6, 6, 1, 0, 1, 3, 0, 256, 100, 128, 119, 0, 0] # group_idx float16
case13_params = [40, 20, 112, 112, 18, 224, 0, 2560, 1968, 128, 115, 0, 0] # group_idx float32
case14_params = [40, 20, 112, 112, 18, 224, 0, 2560, 1968, 128, 114, 0, 0] # group_idx float16 perf
case15_params = [40, 20, 112, 112, 18, 224, 0, 2560, 1968, 128, 114, 0, 0] # group_idx bfloat16 perf
case16_params = [40, 20, 100, 100, 2, 200, 0, 2560, 200, 128, 114, 0, 0] # group_idx bfloat16 perf with use ub sum
case17_params = [40, 20, 100, 100, 2, 200, 0, 2560, 200, 128, 114, 0, 0] # group_idx float16 perf with use ub sum
case18_params = [40, 20, 100, 100, 2, 200, 0, 2560, 200, 128, 114, 0, 0] # group_idx float32 perf with use ub sum
case19_params = [20, 20, 1, 0, 1, 1, 0, 2560, 1, 128, 119, 0, 0] # group_idx float16 dimG is 1
case20_params = [20, 20, 1, 0, 1, 1, 0, 2560, 1, 128, 119, 0, 0] # group_idx float32 dimG is 1

# group_idx_type = 1
case21_params = [6, 6, 1, 0, 1, 3, 0, 256, 100, 128, 119, 0, 1]
case22_params = [12, 12, 1, 0, 17, 3, 0, 458, 1968, 128, 119, 0, 1]
case23_params = [40, 20, 100, 100, 2, 200, 0, 2560, 200, 128, 114, 0, 1] # group_idx int64, perf with use ub sum and group_idx_type is 1
case24_params = [40, 20, 100, 100, 2, 200, 0, 2560, 200, 128, 115, 0, 1] # group_idx int32, perf with use ub sum and group_idx_type is 1

params_info = {
    "case0": case0_params,
    "case1": case1_params,
    "case2": case2_params,
    "case3": case3_params,
    "case4": case4_params,
    "case5": case5_params,
    "case6": case6_params,
    "case7": case7_params,
    "case8": case8_params,
    "case9": case9_params,
    "case10": case10_params,
    "case11": case11_params,
    "case12": case12_params,
    "case13": case13_params,
    "case14": case14_params,
    "case15": case15_params,
    "case16": case16_params,
    "case17": case17_params,
    "case18": case18_params,
    "case19": case19_params,
    "case20": case20_params,
    "case21": case21_params,
    "case22": case22_params,
    "case23": case23_params,
    "case24": case24_params,
}


def main():
    case = sys.argv[1]
    print("grouped_bias_add_grad case: ", case)
    tiling = np.array(params_info.get(case), dtype=np.int64)

    first4 = tiling[:4]
    middle = tiling[4:-4]
    last4 = tiling[-4:]

    # 写入文件（前4和后4按uint32_t，中间按int64）
    with open("tiling.bin", "wb") as tiling_file:
        first4.astype(np.uint32).tofile(tiling_file)  # 前4个按uint32存储
        middle.tofile(tiling_file)                    # 中间按int64存储
        last4.astype(np.uint32).tofile(tiling_file)   # 后4个按uint32存储


if __name__ == '__main__':
    main()
