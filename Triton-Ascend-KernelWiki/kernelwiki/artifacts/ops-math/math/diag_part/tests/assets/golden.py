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
        "diag_part": "diag_part_golden"
    }
}


def diag_part_golden(x, **kwargs):
    '''
    Golden function for diag_part.
    All the parameters (names and order) follow @diag_part_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    dtype = x.dtype
    if "bfloat16" in str(dtype):
        x = x.view("float16")
    x_holder = tf.placeholder(x.dtype, shape=x.shape)
    res = tf.diag_part(x_holder)
    with tf.Session() as session:
        res = session.run(res, feed_dict={x_holder: x})
    if "bfloat16" in str(dtype):
        res = res.view(dtype)
    return res
