#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

__golden__ = {"kernel": {"dynamic_block_quant": "dynamic_block_quant_golden"}}


DATA_TYPE_INT_TO_STR = {
    0: "float32",
    1: "float16",
    2: "int8",
    3: "int32",
    4: "uint8",
    6: "int16",
    7: "uint16",
    8: "uint32",
    9: "int64",
    10: "uint64",
    11: "double",
    12: "bool",
    16: "complex64",
    17: "complex128",
    18: "qint8",
    19: "qint16",
    20: "qint32",
    21: "quint8",
    22: "quint16",
    23: "resource",
    25: "dual",
    26: "variant",
    27: "bfloat16",
    29: "int4",
    30: "uint1",
    31: "int2",
    32: "uint2",
    33: "complex32",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    37: "float8_e8m0",
    40: "float4_e2m1",
    41: "float4_e1m2",
    42: "hifloat4",
}


def _get_dtype_range(dt):
    if "bfloat16" in str(dt):
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if "uint4" in str(dt):
        return 0, 15
    if "int4" in str(dt):
        return -8, 7
    if "bool" in str(dt):
        return 0, 1
    if "float4_e2m1" in str(dt):
        return -float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")
    if "float4_e1m2" in str(dt):
        return -float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")
    if "float8_e8m0" in str(dt):
        return float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")
    if "float8_e5m2" in str(dt):
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if "float8_e4m3fn" in str(dt):
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    if "hifloat8" in str(dt):
        return -float.fromhex("0x1.p15"), float.fromhex("0x1.p15")
    if "complex32" in str(dt):
        dt = "float16"
    numpy_dtype = np.dtype(dt)
    if numpy_dtype.kind in "iu":
        numpy_info = np.iinfo(numpy_dtype)
    else:
        numpy_info = np.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max


def dynamic_block_quant_golden(
    x,
    *,
    min_scale=0.0,
    round_mode="rint",
    dst_type=35,
    row_block_size=1,
    col_block_size=128,
    dst_type_max=0.0,
    **kwargs,
):
    """
    Golden function for dynamic_block_quant.
    All the parameters (names and order) follow @dynamic_block_quant_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """

    def replace_inf_nan_with_zero(x):
        x_cleaned = x.copy()
        # 创建一个掩码，标记 inf 和 nan 的位置
        mask = np.isinf(x) | np.isnan(x)

        # 将 inf 和 nan 替换为 0
        x_cleaned[mask] = 0

        # 创建一个与 x 形状相同的张量，inf 和 nan 位置为 1，其余为 0
        mask_tensor = np.zeros_like(x, dtype=int)
        mask_tensor[mask] = 1

        return x_cleaned, mask_tensor

    def block_max_with_padding(x, row_block_size, col_block_size):
        # 获取数组的形状
        rows, cols = x.shape

        # 计算需要填充的行数和列数
        pad_rows = (row_block_size - rows % row_block_size) % row_block_size
        pad_cols = (col_block_size - cols % col_block_size) % col_block_size

        # 对数组进行填充，填充值为 0
        x_padded = np.pad(
            x, ((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0
        )

        # 获取填充后数组的形状
        padded_rows, padded_cols = x_padded.shape

        # 计算每个块的行和列数
        row_blocks = padded_rows // row_block_size
        col_blocks = padded_cols // col_block_size

        # 初始化结果数组
        result = np.zeros((row_blocks, col_blocks))

        # 对每个块进行取最大值操作
        for i in range(row_blocks):
            for j in range(col_blocks):
                # 获取当前块
                block = x_padded[
                    i * row_block_size : (i + 1) * row_block_size,
                    j * col_block_size : (j + 1) * col_block_size,
                ]
                # 取最大值
                result[i, j] = np.max(block)

        return result

    dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
    max_value = 0

    # 计算 max_value 根据 dst_type_str
    if dst_type_str == "hifloat8":
        max_value = pow(2, 15)
    elif dst_type_str == "float8_e5m2":
        max_value = (2 - pow(2, -2)) * pow(2, 15)
    elif dst_type_str == "float8_e4m3fn":
        max_value = (2 - pow(2, -2)) * pow(2, 8)

    # 替换 inf 和 nan 为 0，并获取掩码张量
    x_cleaned, mask_tensor = replace_inf_nan_with_zero(x)

    x_abs = np.abs(x_cleaned)

    # 对数组进行分块并取最大值（包含填充操作）
    block_max = block_max_with_padding(x_abs, row_block_size, col_block_size)

    block_max_f32 = block_max.astype(np.float32)

    # 计算 scale
    if min_scale != 0:
        scale = np.minimum(block_max_f32 / max_value, 1 / min_scale)
    else:
        scale = block_max_f32 / max_value

    # 扩展 scale，使每个块的缩放因子应用到对应的区域
    scale_expanded = np.zeros_like(x_cleaned).astype(np.float32)
    for i in range(scale.shape[0]):
        for j in range(scale.shape[1]):
            scale_expanded[
                i * row_block_size : (i + 1) * row_block_size,
                j * col_block_size : (j + 1) * col_block_size,
            ] = scale[i, j]

    x_f32 = x_cleaned.astype(np.float32)
    out_f32 = x_f32 / scale_expanded
    # 如果x是inf nan 输出还是inf nan
    out_f32[mask_tensor == 1] = x[mask_tensor == 1]
    max_norm = _get_dtype_range(dst_type_str)[1]
    np.clip(out_f32, a_min=-max_norm, a_max=max_norm, out=out_f32)

    output_scale = scale.astype("float32")
    round_data = np.round(out_f32, 8)
    # NPU will cast NaN (with or without sign) to positive ZERO (sign is dropped)
    round_data = np.nan_to_num(round_data, nan=0.0, copy=False)

    if dst_type == 35:
        from ml_dtypes import float8_e5m2

        round_data = round_data.astype(float8_e5m2, copy=False)
    elif dst_type == 36:
        from ml_dtypes import float8_e4m3fn

        round_data = round_data.astype(float8_e4m3fn, copy=False)
    elif dst_type == 34:
        from en_dtypes import hifloat8

        round_data = round_data.astype(hifloat8, copy=False)

    return round_data, output_scale
