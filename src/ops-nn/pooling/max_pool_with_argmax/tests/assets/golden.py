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

import os
import numpy as np
from copy import deepcopy

__golden__ = {"kernel": {"max_pool_with_argmax": "max_pool_with_argmax_golden"}}


def nchw_to_nhwc_ksize_stride(v):
    """
    NCHW: [N, C, H, W]
    NHWC: [N, H, W, C]
    """
    return [v[0], v[2], v[3], v[1]]


def nhwc_index_to_nchw(idx, H, W, C):
    """
    Convert NHWC flatten index to NCHW flatten index
    """
    c = idx % C
    idx //= C
    w = idx % W
    idx //= W
    h = idx % H
    idx //= H
    n = idx
    return ((n * C + c) * H + h) * W + w


def max_pool_with_argmax_golden(
    x,
    *,
    ksize,
    strides,
    padding,
    targmax=9,
    include_batch_in_index=False,
    data_format="NHWC",
    nan_prop=False,
    **kwargs,
):
    """
    Golden function for max_pool_with_argmax.
    All the parameters (names and order) follow @max_pool_with_argmax_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensors (y, argmax)
    """
    os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = "1" if nan_prop else "0"

    import tensorflow as tf

    input_x = deepcopy(x)
    input_format = kwargs["input_formats"][0]
    x_dtype = input_x.dtype

    ksize = list(ksize)
    strides = list(strides)
    include_batch = include_batch_in_index

    # ============================================================
    # 强制 NCHW → NHWC（golden 内部统一 NHWC）
    # ============================================================
    force_nhwc = False

    if input_format == "NCHW" and data_format == "NCHW":
        force_nhwc = True

        input_x = input_x.transpose(0, 2, 3, 1)
        ksize = nchw_to_nhwc_ksize_stride(ksize)
        strides = nchw_to_nhwc_ksize_stride(strides)
        data_format = "NHWC"

    if not force_nhwc:
        if input_format != data_format:
            if input_format == "NHWC" and data_format == "NCHW":
                input_x = input_x.transpose(0, 3, 1, 2)
            elif input_format == "NCHW" and data_format == "NHWC":
                input_x = input_x.transpose(0, 2, 3, 1)
            else:
                raise RuntimeError("Unsupported format conversion")

    x_tf = tf.convert_to_tensor(input_x)

    y_tf, arg_tf = tf.raw_ops.MaxPoolWithArgmax(
        input=x_tf,
        ksize=ksize,
        strides=strides,
        padding=padding,
        Targmax=tf.int64,
        include_batch_in_index=include_batch,
    )

    # ============================================================
    # 输出 dtype
    # ============================================================
    y_np = y_tf.numpy().astype(x_dtype)

    if targmax == 3:
        arg_np = arg_tf.numpy().astype(np.int32)
    else:
        arg_np = arg_tf.numpy().astype(np.int64)

    # ============================================================
    # NCHW index 语义修正
    # ============================================================
    if force_nhwc:
        N, H, W, C = input_x.shape
        arg_flat = arg_np.reshape(-1)
        for i in range(arg_flat.size):
            arg_flat[i] = nhwc_index_to_nchw(int(arg_flat[i]), H, W, C)
        arg_np = arg_flat.reshape(arg_np.shape)

    # ============================================================
    # 输出 format 转换
    # ============================================================
    y_fmt = kwargs["output_formats"][0]
    arg_fmt = kwargs["output_formats"][1]

    if data_format != y_fmt:
        if data_format == "NHWC" and y_fmt == "NCHW":
            y_np = y_np.transpose(0, 3, 1, 2)
        elif data_format == "NCHW" and y_fmt == "NHWC":
            y_np = y_np.transpose(0, 2, 3, 1)

    if data_format != arg_fmt:
        if data_format == "NHWC" and arg_fmt == "NCHW":
            arg_np = arg_np.transpose(0, 3, 1, 2)
        elif data_format == "NCHW" and arg_fmt == "NHWC":
            arg_np = arg_np.transpose(0, 2, 3, 1)

    return y_np, arg_np
