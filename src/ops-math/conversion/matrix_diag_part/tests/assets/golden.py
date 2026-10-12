#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import tensorflow as tf

__golden__ = {"kernel": {"matrix_diag_part": "matrix_diag_part_golden"}}


def matrix_diag_part_golden(x, **kwargs):
    """
    Golden function for matrix_diag_part.
    All the parameters (names and order) follow SE doc prototype definition without outputs.
    All the input Tensors are numpy.ndarray.

    Uses TensorFlow competitor implementation (tf.raw_ops.MatrixDiagPart V1).
    MatrixDiagPart V1 extracts the main diagonal (k=0) of the last two dimensions.
    For input [..., M, N], output is [..., min(M, N)] where output[..., i] = input[..., i, i].

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor: diagonal of shape x.shape[:-2] + [min(M, N)], same dtype as input
    """
    # Record original dtype (pure indexing, no type conversion)
    orig_dtype = x.dtype

    # Use TensorFlow competitor implementation (tf.raw_ops.MatrixDiagPart V1)
    # @constraint: SE doc §6 - Golden must use TF competitor implementation
    tf_x = tf.constant(x)
    tf_result = tf.raw_ops.MatrixDiagPart(input=tf_x)
    result = tf_result.numpy()

    # Ensure output dtype matches input dtype (bit-level exact, no precision loss)
    result = result.astype(orig_dtype, copy=False)

    return result
