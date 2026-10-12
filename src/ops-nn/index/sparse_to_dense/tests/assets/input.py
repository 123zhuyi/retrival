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

__input__ = {"kernel": {"sparse_to_dense": "sparse_to_dense_input"}}


def sparse_to_dense_input(
    indices, output_shape, values, default_value, *, validate_indices=True, **kwargs
):
    """
    Input function for sparse_to_dense.
    All the parameters (names and order) follow @sparse_to_dense_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  input_ranges, full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors (length must match Input count in _def.cpp)
    """
    sparse_indices = indices
    sparse_values = values
    default_value = default_value[0]

    indice_dtype = kwargs["input_dtypes"][0]
    value_dtype = kwargs["input_dtypes"][2]

    default_value = np.array(default_value)
    output_shape = np.array(output_shape)
    sparse_indices = sparse_indices.shape

    indices_num = sparse_indices[0]
    sparse_values_num = len(sparse_values)

    if (sparse_values_num != indices_num) and sparse_values_num == 1:
        sparse_values = np.array(sparse_values[0])

    total_elements = np.prod(output_shape)
    indices_flat = np.random.choice(
        total_elements, size=indices_num, replace=False
    ).astype(indice_dtype)
    sparse_indices = np.unravel_index(indices_flat, output_shape)
    sparse_indices = np.array(sparse_indices).T
    sparse_indices = sparse_indices[np.lexsort(sparse_indices.T[::-1])]

    sparse_indices = sparse_indices.astype(indice_dtype)
    sparse_values = sparse_values.astype(value_dtype)
    default_value = default_value.astype(value_dtype)
    output_shape = output_shape.astype(indice_dtype)

    return [sparse_indices, output_shape, sparse_values, default_value]
