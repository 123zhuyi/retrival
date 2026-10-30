#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

import sys
import numpy as np

def gen_data(dtype, x_shape, mask_shape_type, value):
    """
    生成测试数据：
    - dtype: 数据类型（float32, int32, bfloat16, uint8）
    - x_shape: x的形状（tuple）
    - mask_shape_type: mask形状类型（all_true, all_false, broadcast）
    - value: 填充值
    """
    # 映射数据类型
    dtype_map = {
        "float32": np.float32,
        "int32": np.int32,
        "bfloat16": np.uint16,  # bfloat16用uint16_t存储
        "uint8": np.uint8
    }
    np_dtype = dtype_map[dtype]

    # 生成x（1~N的连续值）
    x = np.arange(1, np.prod(x_shape) + 1, dtype=np_dtype).reshape(x_shape)

    # 生成mask
    if mask_shape_type == "all_true":
        mask = np.ones(x_shape, dtype=np.bool_)
    elif mask_shape_type == "all_false":
        mask = np.zeros(x_shape, dtype=np.bool_)
    elif mask_shape_type == "broadcast":
        # 广播场景：mask形状为(1, x_shape[-1])
        mask = np.array([[True, False, True]], dtype=np.bool_)  # 适配x_shape=(2,3)
    else:
        raise ValueError(f"Invalid mask_shape_type: {mask_shape_type}")

    # 生成value（标量）
    if dtype == "bfloat16":
        # 转换为bfloat16的uint16_t表示
        value_bf16 = np.array([value], dtype=np.float32).view(np.uint16)[0]
        value = np.array([value_bf16], dtype=np.uint16)
    else:
        value = np.array([value], dtype=np_dtype)

    # 生成黄金数据
    golden = np.where(mask, value, x)

    # 保存数据
    x.tofile(f"{dtype}_x.bin")
    mask.tofile(f"{dtype}_mask.bin")
    value.tofile(f"{dtype}_value.bin")
    golden.tofile(f"{dtype}_golden.bin")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 gen_data.py <dtype> <x_dim1> <x_dim2> <mask_shape_type> <value>")
        print("Example: python3 gen_data.py float32 2 3 all_true 10.0")
        sys.exit(1)

    dtype = sys.argv[1]
    x_dim1 = int(sys.argv[2])
    x_dim2 = int(sys.argv[3])
    mask_shape_type = sys.argv[4]
    value = float(sys.argv[5]) if dtype != "int32" else int(sys.argv[5])

    x_shape = (x_dim1, x_dim2) if x_dim2 != 1 else (x_dim1,)
    gen_data(dtype, x_shape, mask_shape_type, value)