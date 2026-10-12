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


__golden__ = {"kernel": {"scatter_nd": "scatter_nd_golden"}}


def scatter_nd_golden(indices, x, shape, **kwargs):
    """
    Golden function for scatter_nd.
    All the parameters (names and order) follow @scatter_nd_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    indices_data = indices
    updates_data = x
    shape_data = shape

    updates_shape = updates_data.shape
    indices_shape = indices_data.shape

    updates = tf.compat.v1.placeholder(dtype=updates_data.dtype, shape=updates_shape)
    indices = tf.compat.v1.placeholder(dtype=indices_data.dtype, shape=indices_shape)
    shape = tf.constant(shape_data, dtype=indices_data.dtype)

    with tf.compat.v1.Session() as sess:
        scatter_res = tf.compat.v1.scatter_nd(indices, updates, shape, name=None)
        res = sess.run(
            scatter_res, feed_dict={updates: updates_data, indices: indices_data}
        )
    return res
