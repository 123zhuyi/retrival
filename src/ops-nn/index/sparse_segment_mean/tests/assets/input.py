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

__input__ = {"kernel": {"sparse_segment_mean": "sparse_segment_mean_input"}}


def sparse_segment_mean_input(x, indices, segment_ids, **kwargs):
    """
    Input function for sparse_segment_mean.
    All the parameters (names and order) follow @sparse_segment_mean_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors
    """
    inputx = np.random.uniform(1, 10, x.shape).astype(np.int32)
    inputx = inputx.astype(x.dtype)
    lower_limit = 0
    upper_limit = kwargs["output_ori_shapes"][0][0]

    x_dim0 = x.shape[0]
    indices = np.random.uniform(lower_limit, x_dim0, indices.shape).astype(
        indices.dtype
    )

    segment_ids = np.random.uniform(lower_limit, upper_limit, segment_ids.shape).astype(
        segment_ids.dtype
    )

    segment_ids = np.sort(segment_ids)
    segment_ids[-1] = upper_limit - 1
    return [inputx, indices, segment_ids]
