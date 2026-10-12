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

import copy
import re

import numpy as np

__golden__ = {
    "kernel": {
        "dynamic_mx_quant_with_dual_axis": "dynamic_mx_quant_with_dual_axis_golden"
    }
}


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


def _numpy_mx_dtype(mx_ele_dtype):
    if mx_ele_dtype == "float4_e2m1":
        from en_dtypes import float4_e2m1

        return float4_e2m1
    if mx_ele_dtype == "float4_e1m2":
        from en_dtypes import float4_e1m2

        return float4_e1m2
    if mx_ele_dtype == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    if mx_ele_dtype == "float8_e5m2":
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    if mx_ele_dtype == "float8_e8m0":
        from en_dtypes import float8_e8m0

        return float8_e8m0
    raise ValueError(f"Unsupported mx dtype: {mx_ele_dtype}")


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


def _mx_reshape_to_blocks(fp_array, axis, block_size):
    fp_array = np.expand_dims(fp_array, axis=axis + 1)
    orig_shape = fp_array.shape
    pad = [[0, 0] for _ in range(len(orig_shape))]
    pad_size = orig_shape[axis] % block_size
    pad[axis][1] = block_size - pad_size
    if pad_size > 0:
        fp_array = np.pad(fp_array, pad, "constant")
    padded_shape = fp_array.shape
    reshape = list(padded_shape)
    reshape[axis + 1] = block_size
    reshape[axis] = reshape[axis] // block_size
    fp_array = fp_array.reshape(reshape)
    return fp_array, orig_shape, padded_shape


def _mx_undo_reshape_to_blocks(fp_array, axis, orig_shape, padded_shape):
    # Undo tile reshaping
    fp_array = fp_array.reshape(padded_shape)
    # Undo padding
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    # Remove extra dimension
    fp_array = np.squeeze(fp_array, axis=axis + 1)
    return fp_array


def _mx_calculate_share_exp(fp_array, scale_axis, mx_ele_dtype):
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    max_norm = _get_dtype_range(mx_ele_dtype)[1]
    ele_emax = int(np.log2(max_norm))
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True)
    res = (
        np.floor(
            np.log2(fp_abs_max.astype(np.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))
        )
        - ele_emax
    )
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_calculate_share_exp_nv(fp_array, scale_axis, mx_ele_dtype):
    max_norm = _get_dtype_range(mx_ele_dtype)[1]
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True).astype(
        np.float32
    )
    s_fp32 = fp_abs_max / max_norm
    binary_ints = np.array(s_fp32.view(np.uint32))
    exponent_mask = np.uint32(0x7F800000)
    mantissa_mask = np.uint32(0x007FFFFF)
    exponents = (binary_ints & exponent_mask) >> 23
    exponents_int16 = exponents.astype(np.int16)
    mantissas = binary_ints & mantissa_mask
    condition_1 = (exponents_int16 > 0) & (exponents_int16 < 254) & (mantissas > 0)
    condition_2 = (exponents_int16 == 0) & (mantissas > 2**22)
    exponents_int16 = np.where(
        (condition_1 | condition_2), exponents_int16 + 1, exponents_int16
    )
    res = (exponents_int16 - 127).astype(np.float32)
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_round_mantissa(fp_array, round_mode):
    if round_mode in ("rint", "even"):  # tie to even(c language rint)
        fp_array = np.rint(fp_array)
    elif round_mode in ("round", "nearest"):  # tie away from zero(c language round).
        sign = np.signbit(fp_array)
        rounded_abs = np.floor(np.abs(fp_array) + np.array([0.5], dtype=fp_array.dtype))
        fp_array = np.where(sign, -rounded_abs, rounded_abs)
    elif round_mode == "floor":  # round to minus infinity(c language floor)
        fp_array = np.floor(fp_array)
    elif round_mode == "ceil":  # round to positive infinity(c language ceil)
        fp_array = np.ceil(fp_array)
    elif round_mode == "trunc":  # round to zero(c language truncation)
        fp_array = np.trunc(fp_array)
    else:
        raise Exception(f"Unrecognized round method {round_mode}")
    return fp_array


def _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode):
    mx_dtype = str(mx_ele_dtype)
    match = re.search(r"e(\d+)m(\d+)", mx_dtype)
    if match:
        exp_bits = int(match.group(1))
        mantissa_bits = int(match.group(2))
    else:
        raise ValueError(f"mx element dtype [{mx_ele_dtype}] is not recognized.")

    ret = fp_array / (2**share_exp)
    private_exp = np.floor(np.log2(np.abs(ret.astype(np.float32)) + (ret == 0))).astype(
        fp_array.dtype, copy=False
    )
    # The minimum representable exponent
    min_exp = 0 if "float4_e1m2" in mx_dtype else -(2 ** (exp_bits - 1)) + 2
    private_exp = private_exp.clip(min=min_exp)
    # Scale up so appropriate number of bits are in the integer portion of the number
    ret = ret / (2**private_exp) * (2**mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    # Undo scaling
    ret = ret / (2**mantissa_bits) * (2**private_exp)
    # Set values > max_norm to Inf if desired, else clamp them
    max_norm = _get_dtype_range(mx_dtype)[1]
    np.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
    return ret


def _pad_to_even(tensor, axis):
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions."
        )

    shape = tensor.shape
    length = shape[axis]

    if length % 2 == 0:
        return tensor

    pad_width = [(0, 0)] * tensor.ndim
    pad_width[axis] = (0, 1)

    padded_tensor = np.pad(tensor, pad_width, mode="constant", constant_values=2**-127)
    return padded_tensor


def _interleave(tensor, axis, n_group=2):
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions."
        )
    length = tensor.shape[axis]
    if length % n_group != 0:
        raise ValueError(
            f"Axis length ({length}) must be divisible by n_group ({n_group})"
        )

    group_length = length // n_group
    shape = list(tensor.shape)

    new_shape = shape[:axis] + [group_length, 2] + shape[axis + 1 :]
    reshaped = tensor.reshape(new_shape)

    transpose_order = (
        list(range(0, axis + 1))
        + list(range(axis + 2, len(new_shape)))
        + [
            axis + 1,
        ]
    )

    transposed = reshaped.transpose(transpose_order)
    return transposed


def _mx_quantize(
    fp_array,
    mx_ele_dtype="float4_e2m1",
    axis=-1,
    block_size=32,
    round_mode="rint",
    scale_alg=0,
):
    if not isinstance(fp_array, np.ndarray):
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

    axis = len(fp_array.shape) + axis if axis < 0 else axis
    # padding & reshape to block_size
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(
        fp_array, axis, block_size
    )
    # get mx scale exponents
    if scale_alg == 0 or (mx_ele_dtype in ("float4_e2m1", "float4_e1m2")):
        share_exp = _mx_calculate_share_exp(
            fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype
        )
    else:
        share_exp = _mx_calculate_share_exp_nv(
            fp_array, scale_axis=axis + 1, mx_ele_dtype=mx_ele_dtype
        )
    scale_emax = 2 ** (8 - 1) - 1  # 8 for E8M0
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax

    # quantize mx element
    ele_array = _mx_quantize_to_element_format(
        fp_array, share_exp, mx_ele_dtype, round_mode
    )
    # undo reshape
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = np.squeeze(share_exp, axis=axis + 1)
    # convert to fp8_e8m0 & fp4/fp8 dtype
    ele_dtype_np = _numpy_mx_dtype(mx_ele_dtype)
    # share_exp is always float32
    scale_array = 2**share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)

    # NPU will cast NaN (with or without sign) to positive ZERO (sign is dropped)
    ele_array = np.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)
    # Cube only supports even scales. need to pad zero.
    scale_array_pad = _pad_to_even(scale_array, axis=axis)

    result_shape = copy.deepcopy(list(scale_array_pad.shape))
    result_shape.append(2)

    result_shape[axis] = scale_array_pad.shape[axis] // 2
    # when axis is -1, do not need interleave
    if axis != (len(fp_array.shape) - 1):
        scale_array_pad = _interleave(scale_array_pad, axis=axis)
    scale_array_pad = scale_array_pad.reshape(result_shape)

    scale_array = scale_array_pad.astype(_numpy_mx_dtype("float8_e8m0"), copy=False)

    return scale_array, ele_array


def dynamic_mx_quant_with_dual_axis_golden(
    x, *, round_mode="rint", dst_type=40, scale_alg=0, dst_type_max=0.0, **kwargs
):
    """
    Golden function for dynamic_mx_quant_with_dual_axis.
    All the parameters (names and order) follow @dynamic_mx_quant_with_dual_axis_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    block_size = 32
    dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
    ret1 = _mx_quantize(
        x,
        mx_ele_dtype=dst_type_str,
        axis=-1,
        block_size=block_size,
        round_mode=round_mode,
        scale_alg=scale_alg,
    )
    ret2 = _mx_quantize(
        x,
        mx_ele_dtype=dst_type_str,
        axis=-2,
        block_size=block_size,
        round_mode=round_mode,
        scale_alg=scale_alg,
    )
    return ret1[1], ret1[0], ret2[1], ret2[0]
