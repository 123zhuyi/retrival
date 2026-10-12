#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT)
# Licensed under CANN Open Software License Agreement Version 2.0

import sys
import numpy as np
import os

def compare_data(d_type):
    dtype_map = {
        "float32": np.float32,
        "int32": np.int32,
        "bfloat16": np.uint16,
        "float16": np.float16
    }
    np_type = dtype_map[d_type]
    
    # 读取数据
    output = np.fromfile(f"{d_type}_output.bin", dtype=np_type)
    golden = np.fromfile(f"{d_type}_golden.bin", dtype=np_type)
    
    # 比较（整数类型精确匹配，浮点类型允许误差）
    if d_type in ["int32"]:
        match = np.array_equal(output, golden)
    else:
        match = np.allclose(output, golden, rtol=1e-3, atol=1e-3)
    
    if match:
        print("PASSED!")
        return True
    else:
        print("FAILED!")
        # 打印前5个不匹配项
        diff_idx = np.where(output != golden)[0][:5]
        for idx in diff_idx:
            print(f"index {idx}: output={output[idx]}, golden={golden[idx]}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 compare_data.py <d_type>")
        exit(1)
    ret = compare_data(sys.argv[1])
    exit(0 if ret else 1)