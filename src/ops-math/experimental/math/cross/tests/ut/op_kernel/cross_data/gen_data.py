#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np
import torch


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return shape_list


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {
        "float32": (np.float32, torch.float32),
        "float16": (np.float16, torch.float16),
        "int32": (np.int32, torch.int32),
        "int16": (np.int16, torch.int16),
        "int8": (np.int8, torch.int8),
        "uint8": (np.uint8, torch.uint8)
    }
    
    if d_type not in d_type_dict:
        raise ValueError(f"Unsupported dtype: {d_type}")
    
    np_type, torch_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)

    # 生成随机输入数据
    if np_type in [np.int8, np.uint8, np.int16, np.int32]:
        input_x1 = np.random.randint(-10, 10, shape).astype(np_type)
        input_x2 = np.random.randint(-10, 10, shape).astype(np_type)
    else:
        input_x1 = np.random.uniform(-10, 10, shape).astype(np_type)
        input_x2 = np.random.uniform(-10, 10, shape).astype(np_type)

    # 使用 torch.linalg.cross 计算 golden
    # 找到维度为3的维度作为叉积维度
    dim = None
    for i, s in enumerate(shape):
        if s == 3:
            dim = i
            break
    
    if dim is None:
        raise ValueError(f"Shape {shape} must have at least one dimension with size 3")
    
    # 转换为 torch tensor 并计算叉积
    torch_x1 = torch.from_numpy(input_x1)
    torch_x2 = torch.from_numpy(input_x2)
    
    golden_torch = torch.linalg.cross(torch_x1, torch_x2, dim=dim)
    golden = golden_torch.numpy().astype(np_type)

    input_x1.astype(np_type).tofile(f"{d_type}_input_t1_cross.bin")
    input_x2.astype(np_type).tofile(f"{d_type}_input_t2_cross.bin")
    golden.astype(np_type).tofile(f"{d_type}_golden_t_cross.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
