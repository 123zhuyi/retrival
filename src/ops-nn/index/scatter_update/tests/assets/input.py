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

__input__ = {"kernel": {"scatter_update": "scatter_update_input"}}


def scatter_update_input(var, indices, updates, *, use_locking=False, **kwargs):
    """
    Input function for scatter_update.
    All the parameters (names and order) follow @scatter_update_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  input_ranges, full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors (length must match Input count in _def.cpp)
    """
    shape_var = var.shape
    shape_indices, dtype_indices, size_indices = (
        indices.shape,
        indices.dtype,
        indices.size,
    )
    max_indices = shape_var[0]

    if var.size * indices.size * updates.size == 0:
        return [var, indices, updates]

    if size_indices > max_indices:
        indices = np.random.choice(max_indices, size_indices, replace=True).astype(
            dtype_indices
        )
    else:
        indices = np.random.choice(max_indices, size_indices, replace=False).astype(
            dtype_indices
        )
    indices = np.reshape(indices, shape_indices)
    return [var, indices, updates]
