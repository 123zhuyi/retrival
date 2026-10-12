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

__input__ = {"kernel": {"sparse_slice": "sparse_slice_input"}}


def sparse_slice_input(indices, values, shape, start, size, **kwargs):
    """
    Input function for sparse_slice.
    All the parameters (names and order) follow @sparse_slice_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  input_ranges, full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors
    """
    input_indices = indices
    nnz, rank = input_indices.shape

    shape_np = np.array(shape, dtype=np.int64)
    coords = set()
    while len(coords) < nnz:
        coord = tuple(np.random.randint(0, shape_np[i]) for i in range(rank))
        coords.add(coord)
    indices_np = np.array(list(coords), dtype=np.int64)
    start_np = np.array(start, dtype=np.int64)
    size_np = np.array(size, dtype=np.int64)

    return [indices_np, values, shape_np, start_np, size_np]
