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

import ast
import numpy as np
from copy import deepcopy


__golden__ = {"kernel": {"adaptive_avg_pool2d_grad": "adaptive_avg_pool2d_grad_golden"}}


def _parse_if_str(value):
    if isinstance(value, str):
        text = value.strip()
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return value
    return value


def _normalize_list_int(value, attr_name):
    value = _parse_if_str(value)

    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, tuple):
        value = list(value)

    if not isinstance(value, list):
        raise RuntimeError(
            f"{attr_name} should be list_int, but got {type(value)}: {value}"
        )

    try:
        value = [int(v) for v in value]
    except (TypeError, ValueError) as err:
        raise RuntimeError(f"{attr_name} should be list_int, but got {value}") from err

    if len(value) not in (3, 4):
        raise RuntimeError(f"{attr_name} only supports 3D/4D shape, but got {value}")

    for dim in value:
        if dim < 0:
            raise RuntimeError(
                f"{attr_name} dim should not be negative, but got {value}"
            )

    return value


def _get_input_ori_format(ori_formats):
    ori_formats = _parse_if_str(ori_formats)

    if isinstance(ori_formats, (tuple, list)) and len(ori_formats) > 0:
        return str(ori_formats[0]).upper()
    if isinstance(ori_formats, str) and ori_formats:
        return ori_formats.upper()

    return "NCHW"


def _to_nchw_array(array, ori_format):
    ori_format = str(ori_format).upper()

    if ori_format == "NHWC":
        if array.ndim == 4:
            return array.transpose(0, 3, 1, 2)
        if array.ndim == 3:
            return array.transpose(2, 0, 1)
        raise RuntimeError(
            f"NHWC input_grad only supports 3D/4D, but got ndim={array.ndim}"
        )

    return array


def _from_nchw_array(array, ori_format):
    ori_format = str(ori_format).upper()

    if ori_format == "NHWC":
        if array.ndim == 4:
            return array.transpose(0, 2, 3, 1)
        if array.ndim == 3:
            return array.transpose(1, 2, 0)
        raise RuntimeError(
            f"NHWC output_grad only supports 3D/4D, but got ndim={array.ndim}"
        )

    return array


def _to_nchw_shape(shape, ori_format):
    ori_format = str(ori_format).upper()

    if ori_format == "NHWC":
        if len(shape) == 4:
            return [shape[0], shape[3], shape[1], shape[2]]
        if len(shape) == 3:
            return [shape[2], shape[0], shape[1]]
        raise RuntimeError(
            f"NHWC orig_input_shape only supports 3D/4D, but got {shape}"
        )

    return shape


def _is_fp16_or_bf16(dtype):
    dtype_str = str(dtype).lower()
    return (
        dtype_str in ("float16", "bfloat16")
        or "float16" in dtype_str
        or "bfloat16" in dtype_str
    )


def _cast_back_to_origin_dtype(array, origin_dtype):
    dtype_str = str(origin_dtype).lower()

    if "bfloat16" in dtype_str:
        try:
            import ml_dtypes

            return array.astype(ml_dtypes.bfloat16)
        except Exception:
            return array.astype(np.float32)

    return array.astype(origin_dtype)


def adaptive_avg_pool2d_grad_golden(input_grad, orig_input_shape, **kwargs):
    """
    Kernel golden for adaptive_avg_pool2d_grad.
    All the parameters follow @adaptive_avg_pool2d_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch
    import torch.nn.functional as F

    input_grad = deepcopy(input_grad)
    input_ori_format = _get_input_ori_format(kwargs.get("input_ori_formats"))

    orig_input_shape = _normalize_list_int(orig_input_shape, "orig_input_shape")

    input_grad_nchw = _to_nchw_array(input_grad, input_ori_format)
    orig_input_shape_nchw = _to_nchw_shape(orig_input_shape, input_ori_format)

    if input_grad_nchw.ndim not in (3, 4):
        raise RuntimeError(
            f"AdaptiveAvgPool2dGrad only supports 3D/4D input_grad, "
            f"but got ndim={input_grad_nchw.ndim}, shape={input_grad_nchw.shape}"
        )

    if len(orig_input_shape_nchw) != input_grad_nchw.ndim:
        raise RuntimeError(
            f"input_grad ndim={input_grad_nchw.ndim} does not match "
            f"orig_input_shape={orig_input_shape}"
        )

    if tuple(input_grad_nchw.shape[:-2]) != tuple(orig_input_shape_nchw[:-2]):
        raise RuntimeError(
            f"input_grad non-spatial shape={tuple(input_grad_nchw.shape[:-2])} "
            f"does not match orig_input_shape non-spatial shape={tuple(orig_input_shape_nchw[:-2])}"
        )

    out_size = [int(input_grad_nchw.shape[-2]), int(input_grad_nchw.shape[-1])]
    origin_dtype = input_grad.dtype

    if _is_fp16_or_bf16(origin_dtype):
        x_np = np.zeros(tuple(orig_input_shape_nchw), dtype=np.float32)
        y_grad_np = input_grad_nchw.astype(np.float32)
    else:
        x_np = np.zeros(tuple(orig_input_shape_nchw), dtype=origin_dtype)
        y_grad_np = input_grad_nchw.astype(origin_dtype, copy=False)

    x_t = torch.from_numpy(x_np).requires_grad_(True)
    y_grad_t = torch.from_numpy(y_grad_np)

    y = F.adaptive_avg_pool2d(x_t, out_size)

    if tuple(y.shape) != tuple(y_grad_t.shape):
        raise RuntimeError(
            f"PyTorch adaptive_avg_pool2d output shape={tuple(y.shape)} "
            f"does not match input_grad shape={tuple(y_grad_t.shape)}. "
            f"orig_input_shape={orig_input_shape}, out_size={out_size}"
        )

    y.backward(y_grad_t)

    grad_x = x_t.grad.detach().cpu().numpy()

    if _is_fp16_or_bf16(origin_dtype):
        grad_x = _cast_back_to_origin_dtype(grad_x, origin_dtype)
    else:
        grad_x = grad_x.astype(origin_dtype, copy=False)

    grad_x = _from_nchw_array(grad_x, input_ori_format)

    return grad_x
