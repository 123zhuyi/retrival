#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT)
# Licensed under CANN Open Software License Agreement Version 2.0

import sys
import os
import numpy as np

def gen_data_and_golden(start, stop, num, d_type):
    # 映射数据类型
    dtype_map = {
        "float32": np.float32,
        "int32": np.int32,
        "bfloat16": np.uint16,  # 模拟bfloat16
        "float16": np.float16
    }
    np_type = dtype_map[d_type]
    
    # 生成start和stop（单个元素）
    start_val = np.array([start], dtype=np_type)
    stop_val = np.array([stop], dtype=np_type)
    
    # 生成黄金数据（numpy.linspace）
    golden = np.linspace(start, stop, num, endpoint=True, dtype=np.float32)
    # 若输出类型为整数，需截断
    if d_type in ["int32"]:
        golden = golden.astype(np.int32)
    
    # 保存数据
    start_val.tofile(f"{d_type}_start.bin")
    stop_val.tofile(f"{d_type}_stop.bin")
    golden.tofile(f"{d_type}_golden.bin")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 gen_data.py <start> <stop> <num> <d_type>")
        exit(1)
    # 清理旧文件
    os.system("rm -rf *.bin")
    start = float(sys.argv[1])
    stop = float(sys.argv[2])
    num = int(sys.argv[3])
    d_type = sys.argv[4]
    gen_data_and_golden(start, stop, num, d_type)