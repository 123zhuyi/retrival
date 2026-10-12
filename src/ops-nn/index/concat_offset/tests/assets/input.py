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

__input__ = {"kernel": {"concat_offset": "concat_offset_input"}}


def concat_offset_input(concat_dim, x, *, N, **kwargs):
    """
    Input function for concat_offset.
    All the parameters (names and order) follow @concat_offset_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        x: list of dynamic input tensors
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  input_ranges, full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors
    """
    dyn_input = []

    # dynamic input first is concat_dim
    concat_dim_val = int(concat_dim)
    ipt_tmp = np.array([concat_dim_val], dtype=kwargs["input_dtypes"][0])
    dyn_input.append(ipt_tmp)

    # generate first shape tensor (input_arrays[0] is concat_dim, tensors start at index 1)
    first_tensor = x[0]
    shape_dtype = first_tensor.dtype
    sz = first_tensor.size
    context_input_array0 = np.random.randint(
        low=1, high=np.iinfo(shape_dtype).max, size=(sz,), dtype=shape_dtype
    )
    dyn_input.append(context_input_array0)

    # remaining N-1 shape tensors only change concat_dim position value
    n_shape_tensors = len(x)
    for _ in range(n_shape_tensors - 1):
        tmp = context_input_array0.copy()
        random_dim_num = np.random.randint(low=1, high=np.iinfo(shape_dtype).max)
        tmp[concat_dim_val] = random_dim_num
        dyn_input.append(tmp)

    return dyn_input
