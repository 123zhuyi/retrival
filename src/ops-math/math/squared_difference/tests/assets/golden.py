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
import numpy

__golden__ = {
  	"kernel": {
  	    "squared_difference": "squared_difference_golden"
  	}
}

def squared_difference_golden(x1, x2, **kwargs):
    '''
    Kernel golden for squared_difference.
    All the parameters follow @squared_difference_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
	    input_formats, output_formats, input_ori_formats, output_ori_formats,
	    input_dtypes, output_dtypes.
    '''
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    dtype_name = str(x1.dtype)
    if "bfloat16" in dtype_name or "float16" in dtype_name:
        input0 = tf.compat.v1.placeholder(shape=x1.shape, dtype=tf.float32)
        input1 = tf.compat.v1.placeholder(shape=x2.shape, dtype=tf.float32)
        out = tf.compat.v1.squared_difference(input0, input1)
        feed_dict = {input0: x1.astype("float32"), input1: x2.astype("float32")}
    else:
        input0 = tf.compat.v1.placeholder(shape=x1.shape, dtype=x1.dtype)
        input1 = tf.compat.v1.placeholder(shape=x2.shape, dtype=x2.dtype)
        out = tf.compat.v1.squared_difference(input0, input1, name="squared_difference")
        feed_dict = {input0: x1, input1: x2}
    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        res = sess.run(out, feed_dict=feed_dict)
    return res.astype(kwargs.get("output_dtypes", [res.dtype])[0])
