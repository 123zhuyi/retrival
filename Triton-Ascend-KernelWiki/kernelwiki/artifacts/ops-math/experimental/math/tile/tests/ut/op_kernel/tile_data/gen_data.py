#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import os
import numpy as np


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def gen_data_and_golden(shape_str, multiples_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "int8": np.int8,
        "uint8": np.uint8,
    }
    np_type = d_type_dict.get(d_type, np.float32)
    shape = parse_str_to_shape_list(shape_str)
    multiples = parse_str_to_shape_list(multiples_str)

    np.random.seed(42)
    if np.issubdtype(np_type, np.integer):
        tmp_input = np.random.randint(0, 100, size=shape).astype(np_type)
    else:
        tmp_input = np.random.randn(*shape).astype(np_type)

    tmp_golden = np.tile(tmp_input, multiples)

    tmp_input.tofile(f"{d_type}_input_t_tile.bin")
    tmp_golden.tofile(f"{d_type}_golden_t_tile.bin")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: gen_data.py <shape> <multiples> <dtype>")
        print("Example: gen_data.py '2,3' '3,2' float32")
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])
