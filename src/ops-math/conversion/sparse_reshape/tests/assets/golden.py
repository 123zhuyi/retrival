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

import numpy as np

__golden__ = {
    "kernel": {"sparse_reshape": "sparse_reshape_golden"}
}


def sparse_reshape_golden(indices, shape, new_shape, **kwargs):
    '''
    Golden function for sparse_reshape.
    All the parameters (names and order) follow @sparse_reshape_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        indices: np.ndarray, shape=(nnz, input_rank), dtype=int32/int64
            Non-zero element coordinates
        shape: np.ndarray, shape=(input_rank,), dtype=int32/int64
            Original dense shape
        new_shape: np.ndarray, shape=(output_rank,), dtype=int32/int64
            Target dense shape, allows one dimension to be -1
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Tuple of (y_indices, y_shape)
    '''
    indices = np.asarray(indices)
    shape = np.asarray(shape)
    new_shape = np.asarray(new_shape).copy()

    input_rank = len(shape)
    output_rank = len(new_shape)
    nnz = indices.shape[0]
    dense_size = int(np.prod(shape))

    # Handle -1 dimension
    product = 1
    unknown_index = -1
    for d in range(output_rank):
        if new_shape[d] == -1:
            assert unknown_index == -1, "At most one dimension can be -1"
            unknown_index = d
        else:
            assert new_shape[d] >= 0, f"Dimension {d} must be non-negative"
            product *= int(new_shape[d])

    if unknown_index != -1:
        assert product > 0, "Cannot infer missing dimension with zero product"
        missing = dense_size // product
        assert product * missing == dense_size, \
            f"Input has {dense_size} elements but requested shape requires multiple of {product}"
        new_shape[unknown_index] = missing

    assert int(np.prod(new_shape)) == dense_size, \
        f"Shape mismatch: input has {dense_size} elements, output has {int(np.prod(new_shape))}"

    # Compute y_shape
    y_shape = new_shape.astype(shape.dtype)

    # Compute y_indices
    y_indices = np.zeros((nnz, output_rank), dtype=indices.dtype)

    if nnz > 0:
        # Compute input strides
        input_strides = np.zeros(input_rank, dtype=np.int64)
        if input_rank > 0:
            input_strides[input_rank - 1] = 1
            for d in range(input_rank - 2, -1, -1):
                input_strides[d] = input_strides[d + 1] * int(shape[d + 1])

        # Compute output strides
        output_strides = np.zeros(output_rank, dtype=np.int64)
        if output_rank > 0:
            output_strides[output_rank - 1] = 1
            for d in range(output_rank - 2, -1, -1):
                output_strides[d] = output_strides[d + 1] * int(new_shape[d + 1])

        # For each non-zero element, compute flat index and decompose
        for i in range(nnz):
            flat_id = 0
            for j in range(input_rank):
                flat_id += int(indices[i, j]) * input_strides[j]
            for j in range(output_rank):
                y_indices[i, j] = flat_id // output_strides[j]
                flat_id = flat_id % output_strides[j]

    return y_indices, y_shape
