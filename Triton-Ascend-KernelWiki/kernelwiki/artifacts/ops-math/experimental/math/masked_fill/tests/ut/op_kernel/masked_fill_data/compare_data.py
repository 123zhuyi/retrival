#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

import sys
import numpy as np

def compare_data(dtype):
    """
    比较输出数据与黄金数据：
    - dtype: 数据类型（float32, int32, bfloat16, uint8）
    """
    # 映射数据类型和形状
    dtype_map = {
        "float32": (np.float32, 1e-3),  # 浮点误差容忍
        "int32": (np.int32, 0),         # 整数精确匹配
        "bfloat16": (np.uint16, 0),     # bfloat16用uint16_t精确匹配
        "uint8": (np.uint8, 0)          # uint8精确匹配
    }
    np_dtype, atol = dtype_map[dtype]

    # 读取数据
    output = np.fromfile(f"{dtype}_output.bin", dtype=np_dtype)
    golden = np.fromfile(f"{dtype}_golden.bin", dtype=np_dtype)

    # 验证形状一致
    if output.shape != golden.shape:
        print(f"Shape mismatch: output={output.shape}, golden={golden.shape}")
        return False

    # 验证数值一致
    if dtype == "bfloat16":
        # bfloat16需转换为float32后比较（避免uint16_t直接比较的误差）
        output_float = output.view(np.float32)
        golden_float = golden.view(np.float32)
        match = np.allclose(output_float, golden_float, atol=1e-3)
    else:
        match = np.allclose(output, golden, atol=atol)

    if match:
        print("PASSED!")
        return True
    else:
        print("FAILED!")
        # 打印前5个不匹配项
        diff_idx = np.where(output != golden)[0][:5]
        for idx in diff_idx:
            print(f"Index {idx}: output={output[idx]}, golden={golden[idx]}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 compare_data.py <dtype>")
        sys.exit(1)

    dtype = sys.argv[1]
    ret = compare_data(dtype)
    sys.exit(0 if ret else 1)