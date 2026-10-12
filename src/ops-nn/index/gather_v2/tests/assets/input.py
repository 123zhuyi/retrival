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

__input__ = {"kernel": {"gather_v2": "gather_v2_input"}}


def gather_v2_input(
    x, indices, axis, *, batch_dims=0, negative_index_support=False, **kwargs
):
    """
    Input function for gather_v2.
    All the parameters (names and order) follow @gather_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors
    """
    axis = int(axis) if axis is not None else 0
    idx_zero_rep_rate = kwargs.get("idx_zero_rep_rate", 0)
    if not x.dtype == "bool":
        ipt_0 = np.arange(0, x.size, 1, dtype=x.dtype).reshape(x.shape)
    else:
        ipt_0 = np.random.choice(a=[False, True], size=x.shape, p=[0.5, 0.5]).reshape(
            x.shape
        )
    if idx_zero_rep_rate == 0:
        ipt_1 = np.random.default_rng(0).choice(ipt_0.shape[axis], size=indices.size)
    else:
        ipt_1 = np.random.default_rng(0).choice(
            ipt_0.shape[axis],
            size=indices.size,
            p=[idx_zero_rep_rate]
            + [(1 - idx_zero_rep_rate) / (ipt_0.shape[axis] - 1)]
            * (ipt_0.shape[axis] - 1),
        )
    ipt_1 = ipt_1.astype(indices.dtype)
    ipt_1 = np.reshape(ipt_1.astype(indices.dtype), indices.shape)

    input_dtypes = kwargs["input_dtypes"]
    if len(input_dtypes) >= 3:
        ipt_2 = np.array([axis], dtype=input_dtypes[2])
    else:
        ipt_2 = np.array([0], dtype="int32")
    return [ipt_0, ipt_1, ipt_2]
