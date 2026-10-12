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

import sys
import os
import numpy as np

def parse_str_to_shape_list(shape_str):
    """将字符串shape转换为列表"""
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)

# ="float32"
def gen_data_and_golden(shape_str, d_type, out_shape_str):
    """生成输入数据和golden结果"""
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16
    }
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)
    out_shape = parse_str_to_shape_list(out_shape_str)

    size = np.prod(shape)
    
    # 生成随机测试数据（包含边界值和特殊值）

    tmp_input_x = np.random.choice([-1, -0.5, 0.5, 1, 2], size=size)
    tmp_input_x = tmp_input_x.reshape(shape).astype(np_type)
    
    tmp_golden = np.broadcast_to(tmp_input_x, out_shape).astype(np_type)
    
    # 保存为二进制文件
    tmp_input_x.astype(np_type).tofile(f"{d_type}_input_x_expandv.bin")
    tmp_golden.astype(np_type).tofile(f"{d_type}_golden_expandv.bin")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 gen_data.py '<shape>' '<dtype> <out_shape>'")
        print("Example: python3 gen_data.py '(4, 1 , 3)' 'float32' '(4, 5, 3)'")
        exit(1)
    
    # 清理bin文件
    os.system("rm -rf *.bin")
    # 生成数据
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"Generated test data for shape {sys.argv[1]} and dtype {sys.argv[2]}")