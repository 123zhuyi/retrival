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

__input__ = {"kernel": {"scatter_nd_update": "scatter_nd_update_input"}}


def scatter_nd_update_input(var, indices, updates, *, use_locking=False, **kwargs):
    """
    Input function for scatter_nd_update.
    All the parameters (names and order) follow @scatter_nd_update_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  input_ranges, full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors (length must match Input count in _def.cpp)
    """
    deduplicate_flag = True
    if not deduplicate_flag:
        return [var, indices, updates]
    indice_ranksize = indices.shape[-1]
    indice_num = np.prod(indices.shape[:-1])

    all_coord_shape = var.shape[:indice_ranksize]
    sel_coord_shape = indices.shape[:-1]
    if indice_num > np.prod(var.shape[:indice_ranksize]):
        return [var, indices, updates]

    all_coords = np.indices(all_coord_shape).reshape(indice_ranksize, -1).T
    selected_coords = all_coords[
        np.random.choice(
            all_coords.shape[0], size=np.prod(sel_coord_shape), replace=False
        )
    ]
    res_indices = selected_coords.reshape(indices.shape).astype(indices.dtype)
    return [var, res_indices, updates]
