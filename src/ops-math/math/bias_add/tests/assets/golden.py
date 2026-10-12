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
        "bias_add": "bias_add_golden"
    }
}


def bias_add_golden(x, bias, *, data_format=None, **kwargs):
    '''
    Golden function for bias_add.
    All the parameters (names and order) follow @bias_add_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    x_input = tf.compat.v1.placeholder(shape=x.shape, dtype=x.dtype)
    bias_input = tf.compat.v1.placeholder(shape=bias.shape, dtype=bias.dtype)

    out = tf.nn.bias_add(x_input, bias_input, data_format=data_format, name="biasadd")
    feed_dict = {x_input: x, bias_input: bias}
    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        res = sess.run(out, feed_dict=feed_dict)
    return res.astype(kwargs['output_dtypes'][0])
