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

import numpy


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


def get_dtype_range(dt):
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
    numpy_dtype = numpy.dtype(dt)
    if numpy_dtype.kind in "iu":
        numpy_info = numpy.iinfo(numpy_dtype)
    else:
        numpy_info = numpy.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max


def ensure_en_dtypes_version(version):
    import en_dtypes

    cur_ver = list(map(int, en_dtypes.__version__.split("."))) + [0, 0]
    min_ver = list(map(int, version.split("."))) + [0, 0]
    if (cur_ver[0], cur_ver[1], cur_ver[2]) >= (min_ver[0], min_ver[1], min_ver[2]):
        return
    else:
        raise RuntimeError(f"Please upgrade en-dtypes to at least {version}")


def numpy_float8_e5m2():
    try:
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    except ModuleNotFoundError:
        raise RuntimeError(
            "ml_dtypes is needed to support float8_e5m2 dtype!!! "
            "Please install with `pip3 install ml-dtypes`"
        )


def numpy_float8_e4m3fn():
    try:
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    except ModuleNotFoundError:
        raise RuntimeError(
            "ml_dtypes is needed to support float8_e4m3fn dtype!!! "
            "Please install with `pip3 install ml-dtypes`"
        )


def numpy_hifloat8():
    try:
        from en_dtypes import hifloat8

        ensure_en_dtypes_version("0.0.4")
        return hifloat8
    except ModuleNotFoundError:
        raise RuntimeError(
            "en_dtypes is needed to support hifloat8 dtype!!! "
            "Please install with `pip3 install en-dtypes`"
        )
    except ImportError:
        raise RuntimeError(
            "Please upgrade en_dtypes to v0.0.4 at least to support hifloat8 dtype!!! "
            "Command is `pip3 install --upgrade en-dtypes`"
        )


__golden__ = {
    "kernel": {"grouped_dynamic_block_quant": "grouped_dynamic_block_quant_golden"}
}


def _grouped_block_reshape_to_blocks(
    x_array: numpy.ndarray,
    group_list: numpy.ndarray,
    row_block_size: int,
    col_block_size: int,
):
    """
    根据 group_list 对 x_array 按轴分组，每个组内按 row_block_size 和 col_block_size 对齐补 Pad
    补齐 Pad 之后对每个块取最大值

    Args:
        x_array: 输入数组，输入前已取绝对值处理
        group_list: 分组索引，定义每个组的结束位置
        row_block_size: row方向的块大小
        col_block_size: col方向的块大小

    Returns:
        result: 每个块的最大值
    """
    axis = -2
    # Step 1: Split into groups based on group_list
    groups = []
    groups_scale_addr = []
    prev = 0

    for index, end in enumerate(group_list):
        # 沿 axis 轴切片
        slices = [slice(None)] * x_array.ndim
        slices[axis] = slice(prev, end)
        group = x_array[tuple(slices)]
        groups.append(group)
        groups_scale_addr.append(end // row_block_size + index + 1)
        prev = end

    # groups_scale_addr.append(prev // row_block_size + index + 1)
    # Step 2: Pad each group to align with block_size
    padded_groups = []
    padded_group_row_blocks = []
    for group in groups:
        # Get the shape of the group
        batch, rows, cols = group.shape

        # Calculate the number of rows and columns that need to be filled
        pad_rows = (row_block_size - rows % row_block_size) % row_block_size
        pad_cols = (col_block_size - cols % col_block_size) % col_block_size

        # Fill the array with the value 0
        padded_group = numpy.pad(
            group,
            (
                (0, 0),  # Dimension 0 (batch): No padding at the front or back
                (
                    0,
                    pad_rows,
                ),  # Dimension 1 (rows): no padding at the front, pad_rows padding at the back
                (
                    0,
                    pad_cols,
                ),  # Dimension 2 (columns): no padding at the front, pad_cols columns at the back
            ),
            mode="constant",  # Constant padding
            constant_values=0,  # The fill value is 0
        )
        padded_groups.append(padded_group)
        padded_group_row_blocks.append(int((rows + pad_rows) / row_block_size))

    # Step 3: Concatenate all groups along axis
    padded_array = numpy.concatenate(padded_groups, axis=axis)

    # Step 4：Calculate the maximum value of each block
    padded_batch, padded_rows, padded_cols = padded_array.shape

    row_blocks = padded_rows // row_block_size
    col_blocks = padded_cols // col_block_size

    result = numpy.zeros(
        (padded_batch, row_blocks, col_blocks), dtype=padded_array.dtype
    )

    for k in range(padded_batch):
        for i in range(row_blocks):
            for j in range(col_blocks):
                block = padded_array[
                    k,
                    i * row_block_size : (i + 1) * row_block_size,
                    j * col_block_size : (j + 1) * col_block_size,
                ]
                result[k, i, j] = numpy.max(block)

    return result, groups_scale_addr, padded_group_row_blocks


def reshape_scale_array_to_paded_group(
    scale_array: numpy.ndarray, groups_scale_addr: list, padded_group_row_blocks: list
):
    """
    输入：
        scale_array: scale 原始数据
        groups_scale_addr： scale 分组的地址偏移 [0/row_block_size+0, a/row_block_size+1, b/row_block_size+2, c/row_block_size+3, ......]
        padded_group_row_blocks： 每个 group 在 row 方向上的 scale 数量

    输出：
        padded_scale_matrix: 一个数组，形状为 (groups_scale_addr[-1], padded_col_blocks)
                                 其中只有指定的 groups_scale_addr 位置填充了对应数据，其余为 0。
    """

    batch, rows, cols = scale_array.shape
    group_num = len(padded_group_row_blocks)
    stacked_scale_array_list = []

    zero_row = numpy.full((1, cols), 1)
    for i in range(batch):
        sub_array = scale_array[i]
        sub_array_pre = 0
        sub_sub_array_pre = 0
        sub_sub_array = numpy.array([]).reshape(0, cols)
        for j in range(group_num):
            group_row_blocks = int(padded_group_row_blocks[j])
            scale_addr = groups_scale_addr[j]
            sub_sub_array = numpy.vstack(
                [
                    sub_sub_array,
                    sub_array[sub_array_pre : sub_array_pre + group_row_blocks, :],
                ]
            )

            sub_array_pre = sub_array_pre + group_row_blocks
            diff = scale_addr - group_row_blocks - sub_sub_array_pre
            sub_sub_array_pre = sub_sub_array_pre + group_row_blocks + diff

            for _ in range(diff):
                sub_sub_array = numpy.vstack([sub_sub_array, zero_row])

        stacked_scale_array_list.append(sub_sub_array)

    stacked_scale_array = numpy.stack(stacked_scale_array_list, axis=0)

    return stacked_scale_array


def grouped_block_quantize(
    x_array: numpy.ndarray,
    group_list: numpy.ndarray,
    min_scale: float = 0.0,
    round_mode: str = "rint",
    y_dtype: str = "float8_e5m2",
    row_block_size: int = 1,
    col_block_size: int = 128,
    group_list_type: int = 0,
) -> tuple:
    """
    quantize BFP16/FP16 to FLOAT8_E5M2 / FLOAT8_E4M3 / HIFLOAT8 dtypes
    :parameter x_array: input x numpy array with dtype BFP16/FP16
    :parameter group_list: input groupindex numpy array with dtype INT32

    :parameter min_scale: The minimum value of the scale should be greater than or equal to 0.0
    :parameter round_mode: round mode. support rint/round/hybrid
    :parameter y_type: Target data type, only supports FLOAT8_E5M2 (35)/FLOAT_E4M3 (36)/HIFLOAT8 (34)
    :parameter row_block_size: Quantisation granularity on the M-axis
    :parameter col_block_size: Quantisation granularity on the N-axis
    :parameter group_list_type: The format of group_list is 0 (cumsum) or 1 (count), current is 0

    :return: mx-scale-exponents & mx-elements

    NOTE: Scenarios below should be considered and tested when TRYING to modify this code:
    1. block with only subnormal floats
    2. block with only one nan
    3. block with only one inf or -inf
    """
    if not isinstance(x_array, numpy.ndarray):
        raise RuntimeError(
            f"Input tensor to be quantized should be numpy array. But got {type(x_array)}"
        )
    if x_array.dtype.name not in ("bfloat16", "float16"):
        raise RuntimeError(
            f"Dtype of input tensor to be quantized is not supported: {x_array.dtype.name}"
        )
    if y_dtype not in ("float8_e4m3fn", "float8_e5m2", "hifloat8"):
        raise NotImplementedError(f"Not support {y_dtype} yet!")

    def is_non_reverse_order(arr):
        if len(arr) <= 1:
            return True  # 空数组或单元素数组视为逆序
        diff = numpy.diff(arr)
        return numpy.all(diff >= 0)

    if not is_non_reverse_order(group_list):
        raise RuntimeError("Input tensor group_list should be non-reverse order.")

    if group_list[-1] != x_array.shape[-2]:
        raise RuntimeError(
            "The last element of group_list should match the dimension size of the input x M-axis."
        )

    expend_flag = False
    if x_array.ndim == 2:
        expend_flag = True
        x_array = numpy.expand_dims(x_array, axis=0)

    if x_array.ndim != 3:
        raise RuntimeError(
            f"The dimension of the input array must be 3, but the current x_array.ndim is {x_array.ndim}."
        )

    max_value = 0
    # Calculate max_value based on y_dtype
    if y_dtype == "hifloat8":
        max_value = pow(2, 15)
    elif y_dtype == "float8_e5m2":
        max_value = (2 - pow(2, -2)) * pow(2, 15)
    elif y_dtype == "float8_e4m3fn":
        max_value = (2 - pow(2, -2)) * pow(2, 8)

    x_array_abs = numpy.abs(x_array)

    # padding & reshape to block_size & take the maximum value in blocks
    block_max, groups_scale_addr, padded_group_row_blocks = (
        _grouped_block_reshape_to_blocks(
            x_array_abs, group_list, row_block_size, col_block_size
        )
    )

    block_max_f32 = block_max.astype(numpy.float32)

    # 计算 scale
    if min_scale != 0:
        is_finite = numpy.isfinite(block_max_f32)
        scale_all = block_max_f32 / max_value
        scale = numpy.minimum(scale_all, 1 / min_scale)
        scale = numpy.where(is_finite, scale, scale_all)
    else:
        scale = block_max_f32 / max_value

    scale_expanded = numpy.zeros_like(x_array).astype(numpy.float32)

    for k in range(scale.shape[0]):
        scaleOffsetIds = 0
        scaleOffset = 0
        nxtScaleOffset = group_list[0]
        for i in range(scale.shape[1]):
            for j in range(scale.shape[2]):
                value = scale[k, i, j]
                scale_expanded[
                    k,
                    scaleOffset : min(scaleOffset + row_block_size, nxtScaleOffset),
                    j * col_block_size : (j + 1) * col_block_size,
                ] = value
            scaleOffset = scaleOffset + row_block_size
            if scaleOffset >= nxtScaleOffset:
                scaleOffset = nxtScaleOffset
                if scaleOffsetIds + 1 < len(group_list):
                    nxtScaleOffset = group_list[scaleOffsetIds + 1]
                    scaleOffsetIds += 1

    x_f32 = x_array.astype(numpy.float32)
    out_f32 = x_f32 / scale_expanded

    reshape_paded_scale = reshape_scale_array_to_paded_group(
        scale, groups_scale_addr, padded_group_row_blocks
    )
    output_scale = reshape_paded_scale.astype("float32")

    max_norm = get_dtype_range(y_dtype)[1]
    numpy.clip(out_f32, a_min=-max_norm, a_max=max_norm, out=out_f32)

    round_data = numpy.round(out_f32, 8)
    round_data = numpy.nan_to_num(round_data, nan=0.0, copy=False)

    if y_dtype == "float8_e5m2":
        round_data = round_data.astype(numpy_float8_e5m2(), copy=False)
    elif y_dtype == "float8_e4m3fn":
        round_data = round_data.astype(numpy_float8_e4m3fn(), copy=False)
    elif y_dtype == "hifloat8":
        round_data = round_data.astype(numpy_hifloat8(), copy=False)

    if expend_flag and round_data.shape[0] == 1 and output_scale.shape[0] == 1:
        round_data = numpy.squeeze(round_data, axis=0)
        output_scale = numpy.squeeze(output_scale, axis=0)

    return round_data, output_scale


def grouped_dynamic_block_quant_golden(
    x,
    group_list,
    min_scale=0.0,
    round_mode="rint",
    dst_type=35,
    row_block_size=1,
    col_block_size=128,
    group_list_type=0,
    dst_type_max=0.0,
    **kwargs,
):
    """
    Kernel golden for grouped_dynamic_block_quant.
    All the parameters follow @grouped_dynamic_block_quant_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]

    ret = grouped_block_quantize(
        x,
        group_list,
        min_scale,
        round_mode,
        dst_type_str,
        row_block_size,
        col_block_size,
        group_list_type,
    )

    return ret[0], ret[1]
