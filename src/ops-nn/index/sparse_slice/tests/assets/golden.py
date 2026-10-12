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


__golden__ = {"kernel": {"sparse_slice": "sparse_slice_golden"}}


def sparse_slice_golden(indices, values, shape, start, size, **kwargs):
    """
    Golden function for sparse_slice.
    All the parameters (names and order) follow @sparse_slice_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    indices_holder = tf.compat.v1.placeholder(shape=indices.shape, dtype=indices.dtype)
    values_holder = tf.compat.v1.placeholder(shape=values.shape, dtype=values.dtype)
    dense_shape_holder = tf.compat.v1.placeholder(shape=shape.shape, dtype=shape.dtype)
    start_holder = tf.compat.v1.placeholder(shape=start.shape, dtype=start.dtype)
    size_holder = tf.compat.v1.placeholder(shape=size.shape, dtype=size.dtype)

    out_indices, out_values, out_shape = tf.raw_ops.SparseSlice(
        indices=indices_holder,
        values=values_holder,
        shape=dense_shape_holder,
        start=start_holder,
        size=size_holder,
    )

    feed_dict = {
        indices_holder: indices,
        values_holder: values,
        dense_shape_holder: shape,
        start_holder: start,
        size_holder: size,
    }
    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        res = sess.run((out_indices, out_values, out_shape), feed_dict=feed_dict)

    return res
