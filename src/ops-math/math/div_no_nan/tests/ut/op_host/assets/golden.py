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
        "div_no_nan": "div_no_nan_golden"
    }
}


def div_no_nan_golden(x1, x2, **kwargs):
    '''
    Golden function for div_no_nan.
    All the parameters (names and order) follow @div_no_nan_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    x1_placeholder = tf.compat.v1.placeholder(shape=x1.shape, dtype=x1.dtype)
    x2_placeholder = tf.compat.v1.placeholder(shape=x2.shape, dtype=x2.dtype)

    if x1.dtype == np.int8 or x1.dtype == np.int8:
        x1_fp16 = tf.cast(x1_placeholder, tf.float16)
        x2_fp16 = tf.cast(x2_placeholder, tf.float16)
        out = tf.compat.v1.div_no_nan(x1_fp16, x2_fp16, name="divnonan")
        out = tf.cast(out, tf.int8)
    elif x1.dtype == np.uint8 or x1.dtype == np.uint8:
        x1_fp16 = tf.cast(x1_placeholder, tf.float16)
        x2_fp16 = tf.cast(x2_placeholder, tf.float16)
        out = tf.compat.v1.div_no_nan(x1_fp16, x2_fp16, name="divnonan")
        out = tf.cast(out, tf.uint8)
    elif x1.dtype == np.int32 or x1.dtype == np.int32:
        x1_fp32 = tf.cast(x1_placeholder, tf.float32)
        x2_fp32 = tf.cast(x2_placeholder, tf.float32)
        out = tf.compat.v1.div_no_nan(x1_fp32, x2_fp32, name="divnonan")
        out = tf.cast(out, tf.int32)
    else:
        out = tf.compat.v1.div_no_nan(x1_placeholder, x2_placeholder, name="divnonan")

    feed_dict = {x1_placeholder: x1, x2_placeholder: x2}
    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        res = sess.run(out, feed_dict=feed_dict)

    output_dtypes = kwargs.get('output_dtypes', [None])
    return res.astype(output_dtypes[0]) if output_dtypes[0] else res
