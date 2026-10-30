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


__golden__ = {
    "kernel": {
        "random_standard_normal_v2": "random_standard_normal_v2_golden"
    }
}


def random_standard_normal_v2_golden(shape, offset, dtype: int = 0, seed: int = 0, seed2: int = 0, **kwargs):
    '''
    Kernel golden for random_standard_normal_v2.
    All the parameters follow @random_standard_normal_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    out_shape = shape.tolist() if isinstance(shape, np.ndarray) else list(shape)
    out_dtype = kwargs.get("output_dtypes", ["float32"])[0]

    def random_standard_normal(tf_shape):
        op1 = tf.raw_ops.RandomStandardNormal(shape=tf_shape, dtype=out_dtype, seed=seed, seed2=seed2)
        return op1

    x = tf.compat.v1.placeholder(tf.int64, shape=[None])
    op1 = random_standard_normal(x)

    offset_val = offset[0] if isinstance(offset, np.ndarray) else offset[0]
    out_size = 1
    for num in out_shape:
        out_size *= num

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        if offset_val > 0:
            result1 = sess.run(op1, feed_dict={x: [offset_val / 256]})
        result = sess.run(op1, feed_dict={x: out_shape})

    new_offset = np.array([offset_val + 256 * out_size]).astype(kwargs.get("output_dtypes", [None, "int64"])[1])

    return result, new_offset
