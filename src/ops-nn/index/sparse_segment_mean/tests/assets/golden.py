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

__golden__ = {"kernel": {"sparse_segment_mean": "sparse_segment_mean_golden"}}


def sparse_segment_mean_golden(x, indices, segment_ids, **kwargs):
    """
    Golden function for sparse_segment_mean.
    All the parameters (names and order) follow @sparse_segment_mean_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow as tf

    input_x = x
    input_x_dtpye = x.dtype
    if input_x_dtpye.name == "bfloat16":
        input_x = input_x.astype("float32")

    tf.compat.v1.disable_v2_behavior()
    result = tf.sparse.segment_mean(input_x, indices, segment_ids)
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        output = sess.run(result)
    res_output = output.astype(input_x_dtpye)
    return np.array(res_output)
