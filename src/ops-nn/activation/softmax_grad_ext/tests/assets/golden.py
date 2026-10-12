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


__golden__ = {"kernel": {"softmax_grad_ext": "softmax_grad_ext_golden"}}


def softmax_grad_ext_golden(grad, x1, x2, *, axes=-1, keep_dims=True, **kwargs):
    """
    Golden function for softmax_grad_ext.
    All the parameters (names and order) follow @softmax_grad_ext_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow as tf
    from tensorflow.python.ops import gen_math_ops

    axis = axes
    keep_dims_val = keep_dims
    grad_softmax_data = grad
    softmax_data = x1
    x2_data = x2

    tf.compat.v1.disable_eager_execution()
    softmax_shape = tf.compat.v1.placeholder(
        shape=softmax_data.shape, dtype=softmax_data.dtype
    )
    grad_softmax_shape = tf.compat.v1.placeholder(
        shape=grad_softmax_data.shape, dtype=grad_softmax_data.dtype
    )
    x2_tf = tf.compat.v1.placeholder(shape=x2_data.shape, dtype=x2_data.dtype)

    if grad_softmax_data.dtype.name in ("float16", "bfloat16"):
        softmax_shape = gen_math_ops.cast(softmax_shape, tf.float32)
        grad_softmax_shape = gen_math_ops.cast(grad_softmax_shape, tf.float32)
        x2_tf = gen_math_ops.cast(x2_tf, tf.float32)
    mul0 = gen_math_ops.mul(grad_softmax_shape, softmax_shape)
    sum0 = tf.reduce_sum(mul0, axis=axis, keepdims=keep_dims_val)

    mul1 = gen_math_ops.mul(sum0, softmax_shape)
    mul2 = gen_math_ops.mul(grad_softmax_shape, softmax_shape)
    sub0 = gen_math_ops.sub(mul2, mul1)
    mul3 = gen_math_ops.mul(sub0, x2_tf)

    if grad_softmax_data.dtype.name in ("float16", "bfloat16"):
        mul3 = gen_math_ops.cast(mul3, grad_softmax_data.dtype.name)

    feed_dict = {
        grad_softmax_shape: grad_softmax_data,
        softmax_shape: softmax_data,
        x2_tf: x2_data,
    }
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        res = sess.run(mul3, feed_dict=feed_dict)
    return res
