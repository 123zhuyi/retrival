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
import copy


BFP16_NEEDS_FP32_FOR_NPY = None


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


def numpy_bfloat16():
    try:
        from ml_dtypes import bfloat16
    except ModuleNotFoundError:
        try:
            import tensorflow

            bfloat16 = tensorflow.bfloat16.as_numpy_dtype
        except ModuleNotFoundError:
            raise RuntimeError(
                "ml-dtypes or tensorflow is needed to support bfloat16 dtype!!! "
                "Please install with `pip3 install ml-dtypes` "
                "or `pip3 install tensorflow`"
            )
    global BFP16_NEEDS_FP32_FOR_NPY
    if BFP16_NEEDS_FP32_FOR_NPY is None:
        test_array = numpy.array([1, 2, 3], dtype=bfloat16)
        try:
            numpy.max(test_array)
        except TypeError:
            BFP16_NEEDS_FP32_FOR_NPY = True
        else:
            BFP16_NEEDS_FP32_FOR_NPY = False
    return bfloat16


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


def numpy_float8_e8m0():
    try:
        from en_dtypes import float8_e8m0

        ensure_en_dtypes_version("0.0.4")
        return float8_e8m0
    except ModuleNotFoundError:
        raise RuntimeError(
            "en_dtypes is needed to support float8_e8m0 dtype!!! "
            "Please install with `pip3 install en-dtypes`"
        )


def numpy_float4_e2m1():
    try:
        from en_dtypes import float4_e2m1

        ensure_en_dtypes_version("0.0.4")
        return float4_e2m1
    except ModuleNotFoundError:
        raise RuntimeError(
            "en_dtypes is needed to support float4_e2m1 dtype!!! "
            "Please install with `pip3 install en-dtypes`"
        )


def numpy_float4_e1m2():
    try:
        from en_dtypes import float4_e1m2

        ensure_en_dtypes_version("0.0.4")
        return float4_e1m2
    except ModuleNotFoundError:
        raise RuntimeError(
            "en_dtypes is needed to support float4_e1m2 dtype!!! "
            "Please install with `pip3 install en-dtypes`"
        )


__golden__ = {"kernel": {"dynamic_block_mx_quant": "dynamic_block_mx_quant_golden"}}


def dynamic_block_mx_quant_golden(
    x, round_mode="rint", dst_type=40, scale_alg=0, dst_type_max=0.0, **kwargs
):
    """
    Kernel golden for dynamic_block_mx_quant.
    All the parameters follow @dynamic_block_mx_quant_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    block_size = 32
    dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
    fp_array = x
    mx_ele_dtype = dst_type_str

    if not isinstance(fp_array, numpy.ndarray):
        raise RuntimeError(
            f"Input tensor to be quantized should be numpy array. But got {type(fp_array)}"
        )
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32"):
        raise RuntimeError(
            f"Dtype of input tensor to be quantized is not supported: {fp_array.dtype.name}"
        )
    if mx_ele_dtype not in (
        "float4_e2m1",
        "float4_e1m2",
        "float8_e4m3fn",
        "float8_e5m2",
    ):
        raise NotImplementedError(f"Not support {mx_ele_dtype} yet!")

    def block_max_with_padding(
        x, row_block_size, col_block_size, dst_type_str, scale_alg
    ):
        # 获取数组的形状
        batch, rows, cols = x.shape

        # 计算需要填充的行数和列数
        pad_rows = (row_block_size - rows % row_block_size) % row_block_size
        pad_cols = (col_block_size - cols % col_block_size) % col_block_size

        # 对数组进行填充，填充值为 0
        x_padded = numpy.pad(
            x,
            ((0, 0), (0, pad_rows), (0, pad_cols)),
            mode="constant",
            constant_values=0,
        )

        # 获取填充后数组的形状
        padded_batch, padded_rows, padded_cols = x_padded.shape

        # 计算每个块的行和列数
        row_blocks = padded_rows // row_block_size
        col_blocks = padded_cols // col_block_size

        # 初始化结果数组
        result = numpy.zeros((padded_batch, row_blocks, col_blocks))
        EPSILON = 1e-9  # 用于浮点数比较的容差
        is_dst_max_zero = numpy.isclose(dst_type_max, 0.0, atol=EPSILON)
        is_dst_max_six = numpy.isclose(dst_type_max, 6.0, atol=EPSILON)
        is_dst_max_seven = numpy.isclose(dst_type_max, 7.0, atol=EPSILON)

        # 合并判断：是否为 0.0, 6.0 或 7.0
        is_dst_max_target = is_dst_max_zero or is_dst_max_six or is_dst_max_seven

        # 判断目标类型
        is_float4 = dst_type_str == "float4_e2m1"

        # 组合条件：scale_alg == 2 AND float4 AND (max_val in [0.0, 6.0, 7.0])
        is_condition_alg2 = (scale_alg == 2) and is_float4 and is_dst_max_target

        # 对每个块进行取最大值操作
        for k in range(padded_batch):
            for i in range(row_blocks):
                for j in range(col_blocks):
                    # 获取当前块
                    block = x_padded[
                        k,
                        i * row_block_size : (i + 1) * row_block_size,
                        j * col_block_size : (j + 1) * col_block_size,
                    ]
                    if scale_alg == 0:
                        result[k, i, j] = _mx_calculate_share_exp(block, dst_type_str)
                    elif is_condition_alg2:
                        result[k, i, j] = (
                            _block_mx_calculate_share_exp_dynamic_dtype_range(
                                block, dst_type_max, False
                            )
                        )
                    else:
                        raise ValueError(
                            f"Unsupported configuration: alg={scale_alg}, type={dst_type_str}, max={dst_type_max}"
                        )
        return result

    def pad_to_even(tensor: numpy.ndarray, axis: int) -> numpy.ndarray:
        """
        在指定的 axis 上将 tensor 对齐到偶数长度（即按 2 对齐），不足补 0。

        参数:
            tensor (numpy.ndarray): 输入数组
            axis (int): 要对齐的轴（从 0 开始）

        返回:
            numpy.ndarray: 对齐后的数组
        """
        if not isinstance(tensor, numpy.ndarray):
            raise ValueError("Input must be a numpy ndarray.")
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions."
            )

        shape = tensor.shape
        length = shape[axis]

        # 如果已经是偶数，直接返回原数组
        if length % 2 == 0:
            return tensor

        # 构造 pad_width：仅对目标 axis 补一个 0
        pad_width = [(0, 0)] * tensor.ndim
        pad_width[axis] = (0, 1)  # 在 axis 维度末尾补一个 0

        padded_tensor = numpy.pad(
            tensor, pad_width, mode="constant", constant_values=2**-127
        )
        return padded_tensor

    def _mx_calculate_share_exp(fp_array: numpy.ndarray, mx_ele_dtype: str):
        FP32_EXPONENT_BIAS = 127
        FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
        max_norm = get_dtype_range(mx_ele_dtype)[1]
        ele_emax = int(numpy.log2(max_norm))
        fp_abs_max = numpy.max(numpy.abs(fp_array))
        if fp_abs_max == 0:
            res = -float("inf")
        else:
            res = (
                numpy.floor(
                    numpy.log2(
                        fp_abs_max.astype(numpy.float32)
                        + FP32_MIN_NORMAL * (fp_abs_max == 0)
                    )
                )
                - ele_emax
            )
        return res

    def _block_mx_calculate_share_exp_dynamic_dtype_range(
        fp_array: numpy.ndarray, max_norm: float, subnormal: bool
    ):
        import numpy

        if max_norm == 0.0:
            max_norm = 6.0
        bfloat16_type = numpy_bfloat16()
        fp_abs_max = numpy.max(numpy.abs(fp_array)).astype(bfloat16_type)
        binary_ints = fp_abs_max.view(numpy.uint16).item()
        exponent_mask = numpy.uint16(0x7F80)  # 二进制：0111111110000000
        mantissa_mask = numpy.uint16(0x007F)  # 二进制：0000000001111111
        # 提取指数部分并转换为uint16
        exponents = (binary_ints & exponent_mask) >> 7
        exponents_int16 = exponents.astype(numpy.int16)
        exponents_int16_org = exponents.astype(numpy.int16)
        if exponents_int16_org == 0:
            return -127
        # 提取尾数部分并转换为float
        mantissas = binary_ints & mantissa_mask
        mantissas = mantissas.astype(numpy.uint16)
        threshold = numpy.uint32(0x0040) if max_norm == 6 else numpy.uint32(0x0060)
        condition = mantissas > threshold
        exponents_int16 = numpy.where((condition), exponents_int16 + 1, exponents_int16)
        if exponents_int16_org == 255:
            return float("inf")
        exponents_int16 -= 2
        res = (exponents_int16 - 127).astype(numpy.float32)
        if fp_abs_max == 0:
            res = -float("inf")
        return res

    def _block_mx_quantize_to_element_format(
        fp_array: numpy.ndarray,
        share_exp: numpy.ndarray,
        mx_ele_dtype: str,
        round_mode: str,
    ):
        import re

        mx_dtype = str(mx_ele_dtype)
        match = re.search(r"e(\d+)m(\d+)", mx_dtype)
        if match:
            exp_bits = int(match.group(1))
            mantissa_bits = int(match.group(2))
        else:
            raise ValueError(f"mx element dtype [{mx_ele_dtype}] is not recognized.")
        bfloat16_type = numpy_bfloat16()
        bf16_array = fp_array.astype(bfloat16_type)
        binary_ints = numpy.array(bf16_array.view(numpy.uint16))
        exponent_mask = numpy.uint16(0x7F80)  # 二进制：0111111110000000
        # 提取指数部分并转换为uint16
        exponents = (binary_ints & exponent_mask) >> 7
        condition = exponents == 0
        ret = fp_array / (2**share_exp)
        if fp_array.dtype.name == "bfloat16":
            ret = ret.astype(bfloat16_type)
        ret = numpy.where(condition, numpy.where(numpy.signbit(ret), -0.0, 0.0), ret)
        private_exp = numpy.floor(
            numpy.log2(numpy.abs(ret.astype(numpy.float32)) + (ret == 0))
        ).astype(fp_array.dtype, copy=False)
        # The minimum representable exponent
        min_exp = 0 if "float4_e1m2" in mx_dtype else -(2 ** (exp_bits - 1)) + 2
        private_exp = private_exp.clip(min=min_exp)
        # Scale up so appropriate number of bits are in the integer portion of the number
        ret = ret / (2**private_exp) * (2**mantissa_bits)
        ret = _block_mx_round_mantissa(ret, round_mode)
        # Undo scaling
        ret = ret / (2**mantissa_bits) * (2**private_exp)
        # Set values > max_norm to Inf if desired, else clamp them
        max_norm = get_dtype_range(mx_dtype)[1]
        numpy.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
        print(ret)
        return ret

    def _block_mx_round_mantissa(fp_array: numpy.ndarray, round_mode: str):
        """
        For example:
        fp_array  = [-4.5, -3.5, -2.5, -2.0, -1.7, -1.5, -1.4, -0.5, -0.2, -0.0, 0.0, 0.2, 0.5, 1.4, 1.5, 1.7, 2.0, 2.5, 3.4, 4.5]
        - rint    = [-4.,  -4.,  -2.,  -2.,  -2.,  -2.,  -1.,  -0.,  -0.,  -0.,  0.,  0.,  0.,  1.,  2.,  2.,  2.,  2.,  3.,  4.]
        - nearest = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  -0.,  0.,  0.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  5.]
        - floor   = [-5.,  -4.,  -3.,  -2.,  -2.,  -2.,  -2.,  -1.,  -1.,  -0.,  0,   0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
        - ceil    = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  -0.,  0.,  1.,  1.,  2.,  2.,  2.,  2.,  3.,  4.,  5.]
        - trunc   = [-4.,  -3.,  -2.,  -2.,  -1.,  -1.,  -1.,  -0.,  -0.,  -0.,  0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  3.,  4.]
        """
        if round_mode in ("rint", "even"):  # tie to even(c language rint)
            fp_array = numpy.rint(fp_array)
        elif round_mode in (
            "round",
            "nearest",
        ):  # tie away from zero(c language round).
            sign = numpy.signbit(fp_array)
            rounded_abs = numpy.floor(
                numpy.abs(fp_array) + numpy.array([0.5], dtype=fp_array.dtype)
            )
            fp_array = numpy.where(sign, -rounded_abs, rounded_abs)
        elif round_mode == "floor":  # round to minus infinity(c language floor)
            fp_array = numpy.floor(fp_array)
        elif round_mode == "ceil":  # round to positive infinity(c language ceil)
            fp_array = numpy.ceil(fp_array)
        elif round_mode == "trunc":  # round to zero(c language truncation)
            fp_array = numpy.trunc(fp_array)
        else:
            raise Exception(f"Unrecognized round method {round_mode}")
        return fp_array

    def interleave(tensor: numpy.ndarray, axis: int, n_group: int = 2) -> numpy.ndarray:
        if not isinstance(tensor, numpy.ndarray):
            raise ValueError("Input must be a numpy ndarray.")
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions."
            )
        # 获取目标轴的长度
        length = tensor.shape[axis]
        # 检查是否可整除
        if length % n_group != 0:
            raise ValueError(
                f"Axis length ({length}) must be divisible by n_group ({n_group})"
            )

        group_length = length // n_group  # 每组长度
        shape = list(tensor.shape)

        # 重塑形状：在目标轴后插入组维度
        new_shape = shape[:axis] + [group_length, 2] + shape[axis + 1 :]
        reshaped = tensor.reshape(new_shape)

        # 构建转置顺序：交换组维度和组内维度
        transpose_order = (
            list(range(0, axis + 1))  # 目标轴之前的维度
            + list(range(axis + 2, len(new_shape)))
            + [
                axis + 1,
            ]
        )  # 后续维度

        # 执行转置
        transposed = reshaped.transpose(transpose_order)

        return transposed

    expend_flag = False
    if x.ndim == 2:
        expend_flag = True
        x = numpy.expand_dims(x, axis=0)

    share_exp = block_max_with_padding(x, 32, 32, dst_type_str, scale_alg)

    scale_emax = 2 ** (8 - 1) - 1  # 8 for E8M0
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax
    scale_array = 2**share_exp
    scale_array = scale_array.astype(numpy_float8_e8m0(), copy=False)

    batch, rows, cols = x.shape

    batch_padded, rows_padded, cols_padded = scale_array.shape

    scale1_expanded = numpy.zeros((batch, rows, cols_padded))
    scale2_expanded = numpy.zeros((batch, rows_padded, cols))

    for b in range(batch):
        for hb in range(rows_padded):
            # 将这个块的最大值复制给块内的所有行
            for wb in range(cols_padded):
                w_start = wb * block_size
                w_end = min(w_start + block_size, cols)
                scale2_expanded[b, hb, w_start:w_end] = scale_array[b, hb, wb]

    for b in range(batch):
        for hb in range(rows_padded):
            h_start = hb * block_size
            h_end = min(h_start + block_size, rows)
            # 将这个块的最大值复制给块内的所有行
            for wb in range(cols_padded):
                scale1_expanded[b, h_start:h_end, wb] = scale_array[b, hb, wb]

    # share_exp转成x的shape，使每个块的缩放因子应用到对应的区域
    exp__expanded = numpy.zeros_like(x).astype(numpy.float32)
    for k in range(share_exp.shape[0]):
        for i in range(share_exp.shape[1]):
            for j in range(share_exp.shape[2]):
                exp__expanded[
                    k,
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ] = share_exp[k, i, j]
    ele_array = _block_mx_quantize_to_element_format(
        x, exp__expanded, dst_type_str, round_mode
    )

    ele_dtype_np = eval(f"numpy_{mx_ele_dtype}()")

    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)

    # convert to fp8_e8m0 & fp4/fp8 dtype
    ele_dtype_np = eval(f"numpy_{dst_type_str}()")
    # NPU will cast NaN (with or without sign) to positive ZERO (sign is dropped)
    ele_array = numpy.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)

    axis1 = -1 + len(x.shape)
    axis2 = -2 + len(x.shape)
    scale1_array_pad = pad_to_even(scale1_expanded, axis=axis1)
    scale2_array_pad = pad_to_even(scale2_expanded, axis=axis2)

    result_shape1 = copy.deepcopy(list(scale1_array_pad.shape))
    result_shape1.append(2)
    result_shape1[axis1] = scale1_array_pad.shape[axis1] // 2
    scale1_array_pad = scale1_array_pad.reshape(result_shape1)

    result_shape2 = copy.deepcopy(list(scale2_array_pad.shape))
    result_shape2.append(2)
    result_shape2[axis2] = scale2_array_pad.shape[axis2] // 2

    scale2_array_pad = interleave(scale2_array_pad, axis=axis2)
    scale2_array_pad = scale2_array_pad.reshape(result_shape2)

    if (
        expend_flag
        and ele_array.shape[0] == 1
        and scale1_expanded.shape[0] == 1
        and scale2_expanded.shape[0] == 1
    ):
        ele_array = numpy.squeeze(ele_array, axis=0)
        scale1_array_pad = numpy.squeeze(scale1_array_pad, axis=0)
        scale2_array_pad = numpy.squeeze(scale2_array_pad, axis=0)

    scale1 = scale1_array_pad.astype(numpy_float8_e8m0(), copy=False)
    scale2 = scale2_array_pad.astype(numpy_float8_e8m0(), copy=False)
    return ele_array, scale1, scale2
