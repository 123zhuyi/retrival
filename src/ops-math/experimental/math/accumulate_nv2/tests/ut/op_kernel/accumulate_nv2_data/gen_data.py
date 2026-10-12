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
import sys
import os
import re
import numpy as np


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "int8": np.int8,
        "uint8": np.uint8,
    }
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)
    size = np.prod(shape)
    
    input_x = np.random.uniform(-10, 10, shape).astype(np_type)
    golden = np.zeros(shape[1:]).astype(np_type)
    for i in range(shape[0]):
        if d_type in ["float32", "int32"]:
            input_x[i].astype(np_type).tofile(f"{d_type}_input_accumulate_nv2_x{i + 1}.bin")
            golden = input_x[i] + golden
        elif d_type in ["int8", "uint8"]:
            input_x[i].astype(np_type).tofile(f"{d_type}_input_accumulate_nv2_x{i + 1}.bin")
            golden = input_x[i].astype(np.float16) + golden.astype(np.float16)
        else:
            input_x[i].astype(np_type).tofile(f"{d_type}_input_accumulate_nv2_x{i  + 1}.bin")
            golden = input_x[i].astype(np.float32) + golden.astype(np.float32)
    golden.astype(np_type).tofile(f"{d_type}_golden_accumulate_nv2.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])