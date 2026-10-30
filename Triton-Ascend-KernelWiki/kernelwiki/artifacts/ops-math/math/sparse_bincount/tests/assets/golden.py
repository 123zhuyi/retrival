#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

__golden__ = {"kernel": {"sparse_bincount": "sparse_bincount_golden"}}


def sparse_bincount_golden(
    indices, values, dense_shape, size, weights, *, binary_output=False, **kwargs
):
    size_scalar = size.flat[0]
    if values.dtype == np.int64:
        size_scalar = np.int64(size_scalar)
    else:
        size_scalar = np.int32(size_scalar)

    out = tf.raw_ops.SparseBincount(
        indices=tf.constant(indices),
        values=tf.constant(values),
        dense_shape=tf.constant(dense_shape),
        size=tf.constant(size_scalar),
        weights=tf.constant(weights),
        binary_output=binary_output,
    )
    return out.numpy()
