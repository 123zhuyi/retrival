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

__input__ = {"kernel": {"embedding_dense_grad_v2": "embedding_dense_grad_v2_input"}}


def embedding_dense_grad_v2_input(
    grad,
    sort_indices,
    pos_idx,
    *,
    num_weights,
    padding_idx=-1,
    scale_grad_by_freq=False,
    **kwargs,
):
    """
    Input function for embedding_dense_grad_v2.
    All the parameters (names and order) follow @embedding_dense_grad_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors
    """
    index_shape = sort_indices.shape

    sort_indices = sort_indices.reshape(-1)
    sorted_idx = np.sort(sort_indices, axis=-1, kind="stable")
    pos_idx = np.argsort(sort_indices, axis=-1, kind="stable")

    sort_indices = sorted_idx.reshape(index_shape)
    pos_idx = pos_idx.reshape(index_shape)

    return [grad, sort_indices, pos_idx]
