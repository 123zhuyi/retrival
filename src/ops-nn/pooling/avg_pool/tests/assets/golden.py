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


__golden__ = {"kernel": {"avg_pool": "avg_pool_golden"}}


def due_fp16_overflow(data):
    """Overflow interception"""
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    return data


def avg_pool_golden(x, ksize, strides, padding, data_format="NHWC", **kwargs):
    """
    Kernel golden for avg_pool.
    All the parameters follow @avg_pool_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    output_dtype = kwargs.get("output_dtypes")
    # input
    input_arrays = (x,)
    if len(input_arrays) != 4:
        x = input_arrays[0]
        input_dtype = x.dtype
        if "float16" in str(input_dtype):
            x = x.astype(np.float32)

        if len(ksize) == 1:
            ksize = [1, ksize[0], ksize[0], 1]
        if len(ksize) == 2:
            ksize = [1, ksize[0], ksize[1], 1]
        if len(strides) == 1:
            strides = [1, strides[0], strides[0], 1]
        if len(strides) == 2:
            strides = [1, strides[0], strides[1], 1]

        if data_format == "NCHW":
            x = np.transpose(x, (0, 2, 3, 1))
            ksize = [ksize[0], ksize[2], ksize[3], ksize[1]]
            strides = [strides[0], strides[2], strides[3], strides[1]]

        h_index = 1
        w_index = 2
        strideh, stridew = strides[h_index], strides[w_index]
        ksize_h, ksize_w = ksize[h_index], ksize[w_index]
        tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
        avg_pool_result = tf.compat.v1.nn.avg_pool(
            tensor_x,
            ksize=[1, ksize_h, ksize_w, 1],
            strides=[1, strideh, stridew, 1],
            padding=padding,
            data_format="NHWC",
        )
        feed_dict = {tensor_x: x}
        with tf.compat.v1.Session() as sess:
            init_op = tf.compat.v1.global_variables_initializer()
            sess.run(init_op)
            # Generate output tf data
            out = sess.run(avg_pool_result, feed_dict=feed_dict)
        if data_format == "NCHW":
            out = np.transpose(out, (0, 3, 1, 2))
        return out.astype(input_dtype, copy=False)
    x, conv_filter, assist_matrix, bias = input_arrays
    # 5HD input only
    if len(x.shape) != 5:
        raise RuntimeError(
            "avgpool testcase golden function supports NC1HWC0 input only!"
        )
    # Collect shape info
    print("[WARNING] avgpool_mul golden func")
    h_index = data_format.index("H")
    w_index = data_format.index("W")
    strideh, stridew = strides[h_index], strides[w_index]
    ksize_h, ksize_w = ksize[h_index], ksize[w_index]
    IN, IC, IH, IW, C0 = x.shape
    ON = IN
    OC = IC * C0
    if padding == "VALID":
        OH = (IH - ksize_h) // strideh + 1
        OW = (IW - ksize_w) // stridew + 1
    else:
        OH = (IH + strideh - 1) // strideh
        OW = (IW + stridew - 1) // stridew
    # x filter to NHWC
    x = x.transpose(0, 2, 3, 1, 4).reshape(IN, IH, IW, IC * C0).astype(np.float32)
    # 5HD to HWCN
    tensor_x = tf.compat.v1.placeholder(x.dtype, shape=x.shape)
    avg_pool_result = tf.compat.v1.nn.avg_pool(
        tensor_x,
        ksize=[1, ksize_h, ksize_w, 1],
        strides=[1, strideh, stridew, 1],
        padding=padding,
        data_format="NHWC",
    )
    feed_dict = {tensor_x: x}
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(avg_pool_result, feed_dict=feed_dict)

    # NHWC to NC1HWC0
    output = out.reshape((ON, OH, OW, OC // C0, C0)).transpose(0, 3, 1, 2, 4)
    if output_dtype[0] == "float16":
        output = due_fp16_overflow(output.astype(np.float16, copy=False))
    return output
