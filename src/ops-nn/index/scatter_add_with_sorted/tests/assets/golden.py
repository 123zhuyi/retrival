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

__golden__ = {"kernel": {"scatter_add_with_sorted": "scatter_add_with_sorted_golden"}}


def scatter_add_with_sorted_golden(
    var, value, sorted_index, pos=None, *, reduction="add", **kwargs
):
    """
    Golden function for scatter_add_with_sorted.
    All the parameters (names and order) follow @scatter_add_with_sorted_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """

    indices = sorted_index.ravel()
    num_indices = indices.size
    var_first_dim = var.shape[0]
    tail_shape = var.shape[1:]
    tail_size = int(np.prod(tail_shape)) if tail_shape else 1

    if value.size == 1:
        scalar = value.item()
        updates = np.full((num_indices,) + tail_shape, scalar, dtype=var.dtype)
    else:
        expected_size = num_indices * tail_size
        if value.size != expected_size:
            raise ValueError(
                f"updates total elements {value.size} != expected {expected_size} "
                f"(num_indices={num_indices} * var tail dims={tail_shape})"
            )
        updates = value.reshape((num_indices,) + tail_shape).astype(var.dtype)

    if pos is not None:
        update_indices = pos.ravel()
    else:
        update_indices = np.arange(num_indices)

    valid_mask = (indices >= 0) & (indices < var_first_dim)
    valid_var_indices = indices[valid_mask].astype(np.intp)
    valid_update_indices = update_indices[valid_mask].astype(np.intp)
    valid_updates = updates[valid_update_indices]
    out = var.copy()
    np.add.at(out, valid_var_indices, valid_updates)
    return out
