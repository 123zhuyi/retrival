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


__golden__ = {"kernel": {"avg_pool_grad": "avg_pool_grad_golden"}}


def due_fp16_overflow(data):
    """Overflow interception"""
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    return data


def avg_pool_grad_golden(
    orig_input_shape, input_grad, ksize, strides, padding, data_format="NHWC", **kwargs
):
    """
    Kernel golden for avg_pool_grad.
    All the parameters follow @avg_pool_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    output_dtype = kwargs.get("output_dtypes")

    # Collect shape info
    print("[WARNING] AvgpoolGrad golden func")
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    strideh, stridew = strides[h_index], strides[w_index]
    ksize_h, ksize_w = ksize[h_index], ksize[w_index]
    IN, IC, IH, IW, C0 = input_grad.shape
    if data_format == "NCHW":
        N, C, H, W = orig_input_shape
    else:
        N, H, W, C = orig_input_shape
    input_grad = (
        input_grad.transpose(0, 2, 3, 1, 4)
        .reshape(IN, IH, IW, IC * C0)
        .astype(np.float32)
    )
    C = (C + 15) // 16 * 16
    orig_input_shape = (N, H, W, C)
    tensor_dy = tf.compat.v1.placeholder(shape=input_grad.shape, dtype=np.float32)
    avgpoolgrad_result = tf.compat.v1.raw_ops.AvgPoolGrad(
        orig_input_shape=orig_input_shape,
        grad=tensor_dy,
        ksize=[1, ksize_h, ksize_w, 1],
        strides=[1, strideh, stridew, 1],
        padding=padding,
        data_format="NHWC",
        name="avg_pool_grad",
    )
    feed_dict = {tensor_dy: input_grad}
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(avgpoolgrad_result, feed_dict=feed_dict)
    # NHWC to NC1HWC0
    output = out.reshape((N, H, W, C // 16, 16)).transpose(0, 3, 1, 2, 4)
    if output_dtype[0] == "float16":
        output = due_fp16_overflow(output.astype(np.float16, copy=False))
    return output
