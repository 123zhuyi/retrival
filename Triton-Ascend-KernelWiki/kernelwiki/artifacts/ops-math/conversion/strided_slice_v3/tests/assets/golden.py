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
        "strided_slice_v3": "strided_slice_v3_golden"
    }
}


def _construct_valid_axis(axes, ndim):
    axes = np.asarray(axes).flatten().tolist()
    return [a + ndim if a < 0 else a for a in axes]


def _construct_begin_list(begin, ndim, axes, x):
    begin = np.asarray(begin).flatten().tolist()
    full_begin = [0] * ndim
    for i, axis in enumerate(axes):
        b = int(begin[i])
        if b < 0:
            b += x.shape[axis]
        full_begin[axis] = b
    return full_begin


def _construct_end_list(end, ndim, axes, x):
    end = np.asarray(end).flatten().tolist()
    full_end = list(x.shape)
    for i, axis in enumerate(axes):
        e = int(end[i])
        if e < 0:
            e += x.shape[axis]
        full_end[axis] = e
    return full_end


def _construct_stride_list(strides, ndim, axes):
    strides = np.asarray(strides).flatten().tolist()
    full_strides = [1] * ndim
    for i, axis in enumerate(axes):
        full_strides[axis] = int(strides[i])
    return full_strides


def strided_slice_v3_golden(x, begin, end, axes=None, strides=None, **kwargs):
    '''
    Kernel golden for strided_slice_v3.
    All the parameters follow @strided_slice_v3_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    '''
    input_dtype = x.dtype
    if str(input_dtype) == "bfloat16":
        x = x.astype(np.float32)
    elif input_dtype in [np.bool_]:
        x = x.view(np.uint8)

    ndim = x.ndim

    if axes is not None:
        axes_list = _construct_valid_axis(axes, ndim)
    else:
        axes_list = list(range(len(np.asarray(begin).flatten())))

    begin_list = _construct_begin_list(begin, ndim, axes_list, x)
    end_list = _construct_end_list(end, ndim, axes_list, x)
    stride_list = _construct_stride_list(strides, ndim, axes_list) if strides is not None else [1] * ndim

    slices = tuple(slice(begin_list[i], end_list[i], stride_list[i]) for i in range(ndim))
    result = x[slices]

    if str(input_dtype) == "bfloat16":
        result = result.astype(input_dtype)
    elif input_dtype in [np.bool_]:
        result = result.view(np.bool_)

    return result
