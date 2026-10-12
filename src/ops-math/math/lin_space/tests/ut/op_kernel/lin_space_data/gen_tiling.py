#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import sys

case0_params = [1.0, 128.0, 1.0, 128, 128, 4, 32, 32, 301, 0, 0, 0, 0, 0, 0, 0, 0]
case1_params = [1.0, 128.0, 1.0, 128, 128, 4, 32, 32, 301, 0, 0, 0, 0, 0, 0, 0, 0]
case2_params = [1.0, 128.0, 1.0, 128, 128, 4, 32, 32, 301, 0, 0, 0, 0, 0, 0, 0, 0]
case3_params = [1.0, 128.0, 1.0, 128, 128, 4, 32, 32, 301, 0, 0, 0, 0, 0, 0, 0, 0]
case4_params = [1.0, 128.0, 1.0, 128, 128, 4, 32, 32, 301, 0, 0, 0, 0, 0, 0, 0, 0]
case5_params = [1.0, 128.0, 1.0, 128, 128, 4, 32, 32, 301, 0, 0, 0, 0, 0, 0, 0, 0]

params_info = {
    "case0": case0_params,
    "case1": case1_params,
    "case2": case2_params,
    "case3": case3_params,
    "case4": case4_params,
    "case5": case5_params,
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"

    base_params = np.array(params_list, dtype=np.int64)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()