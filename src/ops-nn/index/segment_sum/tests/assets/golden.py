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

__golden__ = {"kernel": {"segment_sum": "segment_sum_golden"}}


def segment_sum_golden(x, segment_ids, **kwargs):
    """
    Golden function for segment_sum.
    All the parameters (names and order) follow @segment_sum_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow as tf

    tf.compat.v1.disable_v2_behavior()

    input_x_dtype = x.dtype
    if input_x_dtype.name == "bfloat16":
        x = x.astype("float32")
    result = tf.math.segment_sum(x, segment_ids)
    with tf.compat.v1.Session() as sess:
        output = sess.run(result)

    return np.array(output).astype(input_x_dtype)
