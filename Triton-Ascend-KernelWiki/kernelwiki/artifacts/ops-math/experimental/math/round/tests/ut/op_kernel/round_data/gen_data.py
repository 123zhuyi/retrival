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
import torch


def parse_str_to_shape_list(shape_str):
    # 解析维度信息，"(2,3,4)"--->array([2, 3, 4])
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def gen_data_and_golden(shape_str, d_type="float32", decimals=0):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": np.float32,
    }
    np_type = d_type_dict.get([d_type]) # 转np格式
    if np_type is None:
        raise ValueError(f"Unsupported data type: {d_type}. Supported types: {list(d_type_dict.keys())}")
    shape = parse_str_to_shape_list(shape_str) # 获取shape数据
    
    # 生成输入和标杆数据
    input_x = np.random.uniform(-100, 100, shape).astype(np_type)
    golden = torch.round(torch.from_numpy(input_x), decimals=decimals).numpy()
    
    # 保存输入和标杆数据
    input_x.astype(np_type).tofile(f"{d_type}_input_round.bin")
    golden.astype(np_type).tofile(f"{d_type}_golden_round.bin")
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])