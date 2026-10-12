#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
#  This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
#  and is contributed to the CANN Open Software.
#  
#  Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
#  All Rights Reserved.
#  
#  Authors (accounts):
#  - Zhou Jianhua<@LePenseur>
#  - Su Tonghua <@sutonghua>
#  
#  This program is free software: you can redistribute it and/or modify it.
#  Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  See the LICENSE file at the root of the repository for the full text of the License.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np

def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "int16": np.int16,
        "int32": np.int32
    }
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)
    size = np.prod(shape)

    # 生成两个随机输入
    input_x = np.random.uniform(-10, 10, size=size).reshape(shape).astype(np_type)
    input_y = np.random.uniform(-10, 10, size=size).reshape(shape).astype(np_type)

    # 计算基准输出
    golden_z = np.subtract(input_x, input_y)

    # 保存文件
    input_x.tofile(f"{d_type}_input_x_sub_v2.bin")
    input_y.tofile(f"{d_type}_input_y_sub_v2.bin")
    golden_z.tofile(f"{d_type}_golden_sub_v2.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 gen_data.py '(shape)' 'dtype'")
        exit(1)
    # 清理旧的bin文件
    os.system("rm -rf ./*.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])



