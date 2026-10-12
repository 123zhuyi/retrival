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


__golden__ = {"kernel": {"unsorted_segment_sum": "unsorted_segment_sum_golden"}}


def unsorted_segment_sum_golden(
    x, segment_ids, num_segments, *, is_preprocessed=False, check_ids=False, **kwargs
):
    """
    Golden function for unsorted_segment_sum.
    All the parameters (names and order) follow @unsorted_segment_sum_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    out = tf.unsorted_segment_sum(x, segment_ids, int(num_segments))
    with tf.Session() as sess:
        res = sess.run(out)
    return [res]
