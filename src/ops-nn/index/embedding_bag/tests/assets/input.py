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

__input__ = {"kernel": {"embedding_bag": "embedding_bag_input"}}


def embedding_bag_input(
    weight,
    indices,
    offsets,
    per_sample_weight=None,
    *,
    mode="mean",
    scale_grad_by_freq=False,
    sparse=False,
    include_last_offset=False,
    padding_idx=-1,
    **kwargs,
):
    """
    Input function for embedding_bag.
    All the parameters (names and order) follow @embedding_bag_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors
    """
    offsets_dtype = offsets.dtype
    if indices.ndim == 1:
        offsets = np.sort(offsets)
        offsets[0] = 0
    else:
        offsets = np.full((indices.shape[0],), indices.shape[1])
        offsets = np.arange(0, indices.size, indices.shape[1])
    if include_last_offset:
        offsets[-1] = len(indices)

    offsets = offsets.astype(offsets_dtype)

    return [weight, indices, offsets, per_sample_weight]
