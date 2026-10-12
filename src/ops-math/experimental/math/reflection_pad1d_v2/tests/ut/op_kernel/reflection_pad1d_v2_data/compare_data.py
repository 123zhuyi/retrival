#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT)
# Licensed under CANN Open Software License Agreement Version 2.0

import sys
import numpy as np
import os

def compare_data(d_type):
    # 数据类型映射
    dtype_map = {
        "float32": np.float32,
        "int32": np.int32,
        "bfloat16": np.uint16,
        "float16": np.float16
    }
    np_type = dtype_map[d_type]
    
    # 读取数据
    output_path = f"{d_type}_output.bin"
    golden_path = f"{d_type}_golden.bin"
    if not os.path.exists(output_path) or not os.path.exists(golden_path):
        print(f"Error: {output_path} or {golden_path} not found!")
        return False
    
    output = np.fromfile(output_path, dtype=np_type)
    golden = np.fromfile(golden_path, dtype=np_type)
    
    # 先检查形状是否一致
    if output.shape != golden.shape:
        print(f"FAILED! Shape mismatch: output={output.shape}, golden={golden.shape}")
        return False
    
    # 比较逻辑
    if d_type in ["int32"]:
        match = np.array_equal(output, golden)
    else:
        # 浮点类型宽松误差，适配硬件计算精度
        match = np.allclose(output, golden, rtol=1e-3, atol=1e-3)
    
    if match:
        print("PASSED!")
        return True
    else:
        print("FAILED!")
        # 打印前5个不匹配项，方便调试
        diff_mask = output != golden
        diff_idx = np.where(diff_mask)[0][:5]
        for idx in diff_idx:
            print(f"index {idx}: output={output[idx]}, golden={golden[idx]}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 compare_data.py <d_type>")
        print("Example: python3 compare_data.py float32")
        exit(1)
    ret = compare_data(sys.argv[1])
    exit(0 if ret else 1)