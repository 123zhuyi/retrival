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

__input__ = {"kernel": {"max_pool_grad_with_argmax": "max_pool_grad_with_argmax_input"}}


def max_pool_grad_with_argmax_input(
    x,
    grad,
    argmax,
    *,
    ksize,
    strides,
    padding,
    include_batch_in_index=False,
    data_format="NHWC",
    **kwargs,
):
    """
    Input function for max_pool_grad_with_argmax.
    All the parameters (names and order) follow @max_pool_grad_with_argmax_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  input_ranges, full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors (length must match Input count in _def.cpp)
    """
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

    input_data = x.numpy() if hasattr(x, "numpy") else x
    dtype = kwargs["input_dtypes"][0]
    argmax_dtype = kwargs["input_dtypes"][2]
    attr_op_format = data_format
    padding = padding

    ksizes = ksize
    strides = strides

    if isinstance(ksizes, int):
        kH, kW = ksizes, ksizes
    elif len(ksizes) == 4:
        if attr_op_format == "NCHW":
            kH, kW = ksizes[2], ksizes[3]
        else:
            kH, kW = ksizes[1], ksizes[2]
    else:
        kH, kW = ksizes[0], ksizes[1]

    if isinstance(strides, int):
        sH, sW = strides, strides
    elif len(strides) == 4:
        if attr_op_format == "NCHW":
            sH, sW = strides[2], strides[3]
        else:
            sH, sW = strides[1], strides[2]
    else:
        sH, sW = strides[0], strides[1]

    tf_ksize = [1, kH, kW, 1]
    tf_strides = [1, sH, sW, 1]

    if attr_op_format == "NCHW":
        N, C, H, W = input_data.shape
        input_nhwc = input_data.transpose(0, 2, 3, 1)
    else:
        N, H, W, C = input_data.shape
        input_nhwc = input_data

    input_ph = tf.placeholder(dtype, shape=input_nhwc.shape)

    grad_nhwc, argmax_nhwc = tf.nn.max_pool_with_argmax(
        input_ph,
        ksize=tf_ksize,
        strides=tf_strides,
        padding=padding,
        data_format="NHWC",
        output_dtype=argmax_dtype,
        include_batch_in_index=include_batch_in_index,
        name="maxpool",
    )

    with tf.Session() as sess:
        grad_val, argmax_val = sess.run(
            (grad_nhwc, argmax_nhwc), feed_dict={input_ph: input_nhwc}
        )

    if attr_op_format == "NCHW":
        grad_out = grad_val.transpose(0, 3, 1, 2)
        N_out, H_out, W_out, C_out = argmax_val.shape
        assert C_out == C

        argmax_nchw = np.zeros((N_out, C_out, H_out, W_out), dtype=argmax_dtype)

        for n in range(N_out):
            for c in range(C_out):
                for ph in range(H_out):
                    for pw in range(W_out):
                        nhwc_index = int(argmax_val[n, ph, pw, c])
                        h_in = nhwc_index // (W * C)
                        remainder = nhwc_index % (W * C)
                        w_in = remainder // C
                        c_in = remainder % C
                        assert c_in == c, f"Channel mismatch: {c_in} vs {c}"

                        local_index = (c_in * H + h_in) * W + w_in
                        argmax_nchw[n, c, ph, pw] = local_index

        argmax_out = argmax_nchw
    else:
        grad_out = grad_val
        argmax_out = argmax_val

    argmax_out = argmax_out.astype(argmax_dtype)

    return [input_data, grad_out, argmax_out]
