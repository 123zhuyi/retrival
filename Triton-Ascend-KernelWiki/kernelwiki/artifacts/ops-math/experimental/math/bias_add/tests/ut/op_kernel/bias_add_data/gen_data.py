#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This file is contributed to the CANN Open Software.
#
# Copyright (c) 2026 Yang Zhenze, Chongqing University of Posts and Telecommunications (CQUPT).
# All Rights Reserved.
#
# Author (account):
# - Yang Zhenze <@gcw_5x5Ew5Ms>
#
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import os
import numpy as np


def gen_data_and_golden(total, channel, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
    }
    np_type = d_type_dict[d_type]
    assert total % channel == 0, "total must be a multiple of channel"

    if d_type == "int32":
        x = np.random.randint(-100, 100, size=total).astype(np_type)
        bias = np.random.randint(-100, 100, size=channel).astype(np_type)
    else:
        x = np.random.uniform(-10, 10, size=total).astype(np_type)
        bias = np.random.uniform(-10, 10, size=channel).astype(np_type)

    bias_broadcast = np.tile(bias, total // channel)
    golden = (x + bias_broadcast).astype(np_type)

    x.tofile(f"{d_type}_input_bias_add_x.bin")
    bias.tofile(f"{d_type}_input_bias_add_bias.bin")
    golden.tofile(f"{d_type}_golden_bias_add.bin")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: gen_data.py <total> <channel> <dtype>")
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
