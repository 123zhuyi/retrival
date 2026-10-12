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

__input__ = {"kernel": {"scatter_nd": "scatter_nd_input"}}


def scatter_nd_input(indices, x, shape, **kwargs):
    """
    Input function for scatter_nd.
    All the parameters (names and order) follow @scatter_nd_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  input_ranges, full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors (length must match Input count in _def.cpp)
    """
    res_shape = np.array(shape, dtype=x.dtype)

    if "id_range" not in kwargs:
        return [indices, x, res_shape]

    indices_range = kwargs["id_range"]
    indices_ranksize = (
        1 if indices.shape == x.shape[: len(indices.shape)] else indices.shape[-1]
    )
    if indices_ranksize != len(indices_range):
        return [indices, x, res_shape]

    for i in range(len(indices_range)):
        low = indices_range[i][0]
        high = indices_range[i][1]
        indices[..., i] = np.random.randint(low, high + 1, size=indices.shape[:-1])

    return [indices, x, res_shape]
