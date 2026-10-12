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
import numpy as np
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
    
    # 为了覆盖 atan2 的四个象限，数据范围取 [-100.0, 100.0]
    low, high = -100.0, 100.0

    if d_type == "bfloat16":
        # 先用 float32 生成再转 bfloat16
        input_x1_f32 = np.random.uniform(low, high, size=shape).astype(np.float32)
        input_x2_f32 = np.random.uniform(low, high, size=shape).astype(np.float32)
        
        input_x1 = input_x1_f32.astype(np_type)
        input_x2 = input_x2_f32.astype(np_type)
        golden = np.arctan2(input_x1_f32, input_x2_f32).astype(np_type)
    elif d_type == "float16":
        # 先生成 float16 数据，用 float32 计算 atan2 保持精度，再转回 float16
        input_x1 = np.random.uniform(low, high, size=shape).astype(np.float16)
        input_x2 = np.random.uniform(low, high, size=shape).astype(np.float16)
        golden = np.arctan2(input_x1.astype(np.float32), input_x2.astype(np.float32)).astype(np.float16)
    else:  # float32
        input_x1 = np.random.uniform(low, high, size=shape).astype(np.float32)
        input_x2 = np.random.uniform(low, high, size=shape).astype(np.float32)
        golden = np.arctan2(input_x1, input_x2).astype(np.float32)

    # 保存 x1, x2 和 golden
    input_x1.astype(np_type).tofile(f"{d_type}_input_x1_t_atan2.bin")
    input_x2.astype(np_type).tofile(f"{d_type}_input_x2_t_atan2.bin")
    golden.astype(np_type).tofile(f"{d_type}_golden_t_atan2.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])