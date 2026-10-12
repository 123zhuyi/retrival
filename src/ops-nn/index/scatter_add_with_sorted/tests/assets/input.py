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

__input__ = {"kernel": {"scatter_add_with_sorted": "scatter_add_with_sorted_input"}}


def scatter_add_with_sorted_input(
    var, value, sorted_index, pos=None, *, reduction="add", **kwargs
):
    indices_flat = sorted_index.ravel()
    order = np.argsort(indices_flat, kind="stable")
    sorted_indices_flat = indices_flat[order]
    orig_shape = sorted_index.shape
    sorted_index_out = sorted_indices_flat.reshape(orig_shape).astype(
        sorted_index.dtype
    )

    if pos is not None:
        pos_out = order.astype(sorted_index.dtype).reshape(orig_shape)
        return (var, value, sorted_index_out, pos_out)
    else:
        num_indices = indices_flat.size
        tail_shape = var.shape[1:]
        value_2d = value.reshape((num_indices,) + tail_shape)
        value_out = value_2d[order].reshape(value.shape).astype(value.dtype)
        return (var, value_out, sorted_index_out)
