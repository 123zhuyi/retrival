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


__input__ = {"kernel": {"avg_pool": "avg_pool_input"}}


def avg_pool_input(x, ksize, strides, padding, data_format="NHWC", **kwargs):
    """
    Input function for avg_pool.
    All the parameters follow @avg_pool_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    """
    output_dtype = kwargs.get("output_dtypes")
    ipt_0 = x
    input_arrays = (x,)
    if len(input_arrays) == 1:
        return (ipt_0,)
    ipt_1 = input_arrays[1]
    ipt_2 = input_arrays[2]
    if input_arrays[1] is None and input_arrays[2] is None:
        return (ipt_0, None, None)
    if len(input_arrays[1].shape) == 4:
        return (ipt_0, ipt_1, None)

    shape = list(ipt_1.shape)
    Co, _, H, W, _ = shape
    Co1 = (Co + 15) // 16
    ipt_1 = numpy.zeros((H * W * Co1, 1, 16, 16), dtype=ipt_1.dtype)
    ipt_1[:, 0, :, :] = numpy.identity(16)
    shape = list(ipt_2.shape)
    n, c1, m, c0 = shape  # mul_shape: [N,Co1,Ho*Wo,C0]
    _, _, in_size_h, in_size_w, _ = list(ipt_0.shape)
    avg_mean_factor = []
    h_index, w_index = data_format.index("H"), data_format.index("W")
    stride_h, stride_w = strides[h_index], strides[w_index]
    window_h, window_w = ksize[h_index], ksize[w_index]
    if padding == "SAME":
        _, _, Hi, Wi, _ = ipt_0.shape
        ho = (Hi + stride_h - 1) // stride_h
        wo = (Wi + stride_w - 1) // stride_w
        pad_rows = max(0, (ho - 1) * stride_h + window_h - in_size_h)
        pad_cols = max(0, (wo - 1) * stride_w + window_w - in_size_w)
        padT = pad_rows // 2
        padL = pad_cols // 2
        avg_mean_factor = []
        for i in range(ho):
            for j in range(wo):
                h_start = i * stride_h - padT
                w_start = j * stride_w - padL
                h_end = min(h_start + window_h, in_size_h)
                w_end = min(w_start + window_w, in_size_w)
                h_start = max(h_start, 0)
                w_start = max(w_start, 0)
                area = max((h_end - h_start) * (w_end - w_start), 1)
                mean_value = int(area)
                avg_mean_factor.append(mean_value)
        avg_mean_factor = numpy.array(avg_mean_factor).astype(output_dtype[0])
    elif padding == "VALID":
        avg_coeff = int(window_h * window_w)
        avg_mean_factor = numpy.random.uniform(avg_coeff, avg_coeff, (m,)).astype(
            output_dtype[0]
        )
    ipt_2 = numpy.zeros((n, c1, m, c0)).astype(output_dtype[0])
    for i in range(c0):
        ipt_2[:, :, :, i] = avg_mean_factor[:]
    return (ipt_0, ipt_1, None)
