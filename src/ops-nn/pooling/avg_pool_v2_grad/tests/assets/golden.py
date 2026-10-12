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


__golden__ = {"kernel": {"avg_pool_v2_grad": "avg_pool_v2_grad_golden"}}


def due_fp16_overflow(data):
    """Overflow interception"""
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    return data


def avg_pool_v2_grad_golden(
    orig_input_shape,
    input_grad,
    ksize,
    strides,
    padding_mode="CALCULATED",
    pads=(0, 0, 0, 0),
    data_format="NCHW",
    global_pooling=False,
    ceil_mode=False,
    exclusive=True,
    divisor_override=0,
    **kwargs,
):
    """
    Kernel golden for avg_pool_v2_grad.
    All the parameters follow @avg_pool_v2_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    output_dtype = kwargs.get("output_dtypes")
    orig_input_shape = [int(x) for x in orig_input_shape]
    ksize = list(ksize)
    strides = list(strides)
    pads = list(pads) if pads is not None else [0, 0, 0, 0]

    if data_format == "NCHW":
        N, C, IH, IW = orig_input_shape
    else:
        N, IH, IW, C = orig_input_shape

    if len(ksize) == 2:
        ksize_h, ksize_w = ksize
    elif len(ksize) == 4:
        ksize_h = ksize[data_format.index("H")]
        ksize_w = ksize[data_format.index("W")]
    else:
        ksize_h = ksize_w = ksize[0]

    if len(strides) == 2:
        stride_h, stride_w = strides
    elif len(strides) == 4:
        stride_h = strides[data_format.index("H")]
        stride_w = strides[data_format.index("W")]
    else:
        stride_h = stride_w = strides[0]

    if global_pooling:
        ksize_h, ksize_w = IH, IW
        stride_h, stride_w = IH, IW
        pads = [0, 0, 0, 0]
        if divisor_override == 0:
            divisor_override = IH * IW

    if ksize_h > IH:
        ksize_h = IH
    if ksize_w > IW:
        ksize_w = IW

    input_dtype = input_grad.dtype
    if data_format == "NCHW":
        OH = input_grad.shape[2]
        OW = input_grad.shape[3]
    else:
        OH = input_grad.shape[1]
        OW = input_grad.shape[2]

    if padding_mode == "CALCULATED":
        pad_top, _, pad_left, _ = pads
    elif padding_mode == "VALID":
        pad_top = 0
        pad_left = 0
    else:
        pad_row = max(0, (OH - 1) * stride_h + ksize_h - IH)
        pad_col = max(0, (OW - 1) * stride_w + ksize_w - IW)
        pad_top = pad_row // 2
        pad_left = pad_col // 2

    if data_format == "NHWC":
        input_grad_nchw = np.transpose(input_grad, (0, 3, 1, 2))
    else:
        input_grad_nchw = input_grad
    input_grad_nchw = input_grad_nchw.astype(np.float32)

    out_grad = np.zeros((N, C, IH, IW), dtype=np.float32)

    for oh in range(OH):
        for ow in range(OW):
            h_start = oh * stride_h - pad_top
            w_start = ow * stride_w - pad_left
            h_end = h_start + ksize_h
            w_end = w_start + ksize_w

            h_start_actual = max(h_start, 0)
            w_start_actual = max(w_start, 0)
            h_end_actual = min(h_end, IH)
            w_end_actual = min(w_end, IW)

            if h_end_actual <= h_start_actual or w_end_actual <= w_start_actual:
                continue

            if divisor_override and divisor_override != 0:
                count = divisor_override
            elif exclusive:
                count = (h_end_actual - h_start_actual) * (
                    w_end_actual - w_start_actual
                )
            else:
                count = ksize_h * ksize_w

            count = max(count, 1)

            out_grad[
                :, :, h_start_actual:h_end_actual, w_start_actual:w_end_actual
            ] += input_grad_nchw[:, :, oh : oh + 1, ow : ow + 1] / count

    if data_format == "NHWC":
        out_grad = np.transpose(out_grad, (0, 2, 3, 1))

    if output_dtype and len(output_dtype) > 0:
        target_dtype = output_dtype[0]
        if target_dtype == "float16":
            out_grad = due_fp16_overflow(out_grad.astype(np.float16, copy=False))
        else:
            out_grad = out_grad.astype(np.dtype(target_dtype), copy=False)
    else:
        out_grad = out_grad.astype(input_dtype, copy=False)

    return out_grad
