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

__golden__ = {"kernel": {"max_pool_grad": "max_pool_grad_golden"}}


def nhwc_2_nc1hwc0(array, dtype):
    shape_from = array.shape
    axis_n = shape_from[0]
    axis_h = shape_from[1]
    axis_w = shape_from[2]
    axis_c = shape_from[3]
    c0 = 32 if dtype == "int8" or dtype == "uint8" else 16
    c1 = (axis_c + c0 - 1) // c0
    x_pad = np.zeros((axis_n, c1 * c0, axis_h, axis_w), dtype)
    tmp_array = array.reshape(shape_from)
    tmp_array = np.transpose(tmp_array, axes=(0, 3, 1, 2))
    x_pad[:, :axis_c, :, :] = tmp_array
    tmp_array = (
        x_pad.reshape(axis_n, c1, c0, axis_h, axis_w).transpose(0, 1, 3, 4, 2).copy()
    )
    return tmp_array


def nc1hwc0_2_nhwc(array, shape_to):
    shape_from = array.shape
    axis_n = shape_from[0]
    axis_c1 = shape_from[1]
    axis_h = shape_from[2]
    axis_w = shape_from[3]
    axis_c0 = shape_from[4]
    c_pad = (
        None if axis_c1 * axis_c0 == shape_to[3] else shape_to[3] - axis_c1 * axis_c0
    )
    array_shape = array.reshape(axis_n, axis_c1, axis_h, axis_w, axis_c0)
    tmp_input_tensor = np.transpose(array_shape, axes=(0, 2, 3, 1, 4))
    tmp_input_tensor = tmp_input_tensor.reshape(
        (axis_n, axis_h, axis_w, axis_c1 * axis_c0)
    )
    return tmp_input_tensor[:, :, :, :c_pad]


def max_pool_grad_golden(
    x1, x2, grad, *, ksize, strides, padding, data_format="NHWC", **kwargs
):
    """
    Golden function for max_pool_grad.
    All the parameters (names and order) follow @max_pool_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    from tensorflow.python.ops import gen_nn_ops

    input_shape, input_orig_y_shape, grad_shape = kwargs["input_ori_shapes"][:3]
    input_data = nc1hwc0_2_nhwc(x1, input_shape)
    input_orig_y_data = nc1hwc0_2_nhwc(x2, input_orig_y_shape)
    grad_data = nc1hwc0_2_nhwc(grad, grad_shape)

    input = tf.compat.v1.placeholder(shape=input_shape, dtype="float16")
    input_orig_y = tf.compat.v1.placeholder(shape=input_orig_y_shape, dtype="float16")
    grad = tf.compat.v1.placeholder(shape=grad_shape, dtype="float16")

    maxpoolgrad_res = gen_nn_ops.max_pool_grad(
        input,
        input_orig_y,
        grad,
        ksize,
        strides,
        padding,
        data_format="NHWC",
        name="maxpoolgrad",
    )
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        res = sess.run(
            maxpoolgrad_res,
            feed_dict={
                input: input_data,
                input_orig_y: input_orig_y_data,
                grad: grad_data,
            },
        )
    res = nhwc_2_nc1hwc0(res, "float16")
    return res.astype(kwargs["output_dtypes"][0], copy=False)
