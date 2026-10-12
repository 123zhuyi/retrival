#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
import re
import tensorflow as tf


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": tf.bfloat16.as_numpy_dtype
    }
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)
    size = np.prod(shape)
    
    input_cond = np.random.choice([True, False], size=size)
    input_cond = input_cond.reshape(shape).astype(np.uint8)

    input_x1 = np.random.uniform(-10, 0, shape).astype(np_type)
    input_x2 = np.random.uniform(0, 10, shape).astype(np_type)

    golden = np.where(condition, input_x1, input_x2).astype(np_type)

    input_cond.astype(np.uint8).tofile(f"{d_type}_input_c_select_v2.bin")
    input_x1.astype(np_type).tofile(f"{d_type}_input_t1_select_v2.bin")
    input_x2.astype(np_type).tofile(f"{d_type}_input_t2_select_v2.bin")
    golden.astype(np_type).tofile(f"{d_type}_golden_t_select_v2.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])



