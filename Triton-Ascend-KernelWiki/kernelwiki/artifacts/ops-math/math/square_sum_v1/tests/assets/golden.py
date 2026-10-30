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


__golden__ = {
    "kernel": {
        "square_sum_v1": "square_sum_v1_golden"
    }
}


def _eliminate_duplicate_axes(axis, input_tensor):
    axis = tuple(set([_ax if _ax >= 0 else len(input_tensor.shape) + _ax for _ax in axis]))
    return axis


def _get_new_format_axis(ori_shape, ori_axis, input_format, ori_format):
    axis = []
    for i in ori_axis:
        new_axis = i % len(ori_shape)
        if new_axis == len(ori_shape) - 1:
            new_axis = [len(ori_shape) - 2, len(ori_shape) + 1]
        elif new_axis == len(ori_shape) - 2:
            new_axis = [len(ori_shape) - 1, len(ori_shape) + 0]
        if isinstance(new_axis, int):
            new_axis = [new_axis]
        axis.extend(list(new_axis))
    return axis


def square_sum_v1_golden(x, axis, keep_dims: bool = False, **kwargs):
    '''
    Kernel golden for square_sum_v1.
    All the parameters follow @square_sum_v1_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    input_format = kwargs.get("input_formats", ["ND"])[0]
    ori_format = kwargs.get("input_ori_formats", ["ND"])[0]
    ori_shape = kwargs.get("input_ori_shapes", [None])[0]
    if input_format == "FRACTAL_NZ" and ori_shape is not None:
        axis = _get_new_format_axis(ori_shape, axis, input_format, ori_format)
    axis_d = []
    if not axis:
        for i, _ in enumerate(x.shape):
            axis_d.append(i)
    else:
        axis_d = axis
    axis_d = _eliminate_duplicate_axes(axis_d, x)
    x_dtype = x.dtype
    if "float16" in str(x_dtype):
        x = x.astype("float32")
    square = np.multiply(x, x)
    res = np.sum(square, axis=axis_d, keepdims=keep_dims)
    if "float16" in str(x_dtype):
        res = res.astype(x_dtype, copy=False)
    return res
