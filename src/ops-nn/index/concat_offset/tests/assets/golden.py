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


__golden__ = {"kernel": {"concat_offset": "concat_offset_golden"}}


def concat_offset_golden(concat_dim, x, *, N, **kwargs):
    """
    Golden function for concat_offset.
    All the parameters (names and order) follow @concat_offset_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        x: list of dynamic input tensors
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow.compat.v1 as tf

    tf.disable_eager_execution()
    from tensorflow.python.ops import gen_array_ops

    concat_dim_val = int(concat_dim)
    out = gen_array_ops.concat_offset(concat_dim_val, x)
    with tf.Session() as sess:
        res = sess.run(out)

    return res
