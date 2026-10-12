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

__input__ = {"kernel": {"scatter_nd_max": "scatter_nd_max_input"}}


def scatter_nd_max_input(var, indices, updates, *, use_locking=False, **kwargs):
    """
    Input function for scatter_nd_max.
    All the parameters (names and order) follow @scatter_nd_max_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    The default generator may produce indices that exceed var's shape (e.g. case
    scatter_nd_max_bool_int64_bool_002 uses indices range (0, 100) while var's first
    two dims are only (13, 15)). Out-of-bounds indices crash ScatterNdMax, so here we
    regenerate indices column-by-column, clamping each index axis d to [0, var.shape[d]).
    """
    index_depth = indices.shape[-1]
    new_indices = np.empty(indices.shape, dtype=indices.dtype)
    for d in range(index_depth):
        upper = var.shape[d]
        if upper <= 0:
            new_indices[..., d] = 0
        else:
            new_indices[..., d] = np.random.randint(0, upper, size=indices.shape[:-1])
    return [var, new_indices, updates]
