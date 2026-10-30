#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT)
# Licensed under CANN Open Software License Agreement Version 2.0

import sys
import os
import numpy as np

def reflection_pad1d_golden(x, pad_left, pad_right):
    """
    计算1D反射填充的黄金数据（与C++核函数逻辑完全一致）
    :param x: 输入一维数组
    :param pad_left: 左侧填充数量
    :param pad_right: 右侧填充数量
    :return: 反射填充后的黄金数据
    """
    w_size = len(x)
    out_w_size = w_size + pad_left + pad_right
    golden = np.zeros(out_w_size, dtype=x.dtype)

    # 左填充：镜像复制，不包含输入左边界
    for i in range(pad_left):
        ref_idx = pad_left - 1 - i  # 与C++核函数索引逻辑对齐
        ref_idx = max(0, min(ref_idx, w_size - 1))  # 边界保护
        golden[i] = x[ref_idx]

    # 中间原始数据
    golden[pad_left:pad_left + w_size] = x

    # 右填充：镜像复制，不包含输入右边界
    for i in range(pad_right):
        ref_idx = w_size - 2 - i  # 与C++核函数索引逻辑对齐
        ref_idx = max(0, min(ref_idx, w_size - 1))  # 边界保护
        golden[pad_left + w_size + i] = x[ref_idx]

    return golden

def gen_data_and_golden(pad_left, pad_right, w_size, d_type):
    # 映射数据类型（与原有逻辑一致）
    dtype_map = {
        "float32": np.float32,
        "int32": np.int32,
        "bfloat16": np.uint16,  # 模拟bfloat16存储
        "float16": np.float16
    }
    np_type = dtype_map[d_type]
    
    # 生成输入x：有序数组（方便验证反射填充效果）
    x = np.arange(w_size, dtype=np.float32).astype(np_type)  # 先生成float32再转换，避免类型异常
    # 生成padding：长度为2的int32数组（[pad_left, pad_right]，与UT一致）
    padding = np.array([pad_left, pad_right], dtype=np.int32)
    
    # 计算黄金数据（反射填充结果）
    golden = reflection_pad1d_golden(x, pad_left, pad_right)
    
    # 保存数据（文件名与UT读取路径完全匹配）
    x.tofile(f"{d_type}_x.bin")
    padding.tofile(f"{d_type}_padding.bin")
    golden.tofile(f"{d_type}_golden.bin")

if __name__ == "__main__":
    # 参数检查：pad_left pad_right w_size d_type（4个参数，与UT调用一致）
    if len(sys.argv) != 5:
        print("Usage: python3 gen_data.py <pad_left> <pad_right> <w_size> <d_type>")
        print("Example: python3 gen_data.py 3 4 8 float32")
        exit(1)
    # 清理旧二进制文件
    os.system("rm -rf *.bin")
    # 解析参数
    pad_left = int(sys.argv[1])
    pad_right = int(sys.argv[2])
    w_size = int(sys.argv[3])
    d_type = sys.argv[4]
    # 生成数据
    gen_data_and_golden(pad_left, pad_right, w_size, d_type)