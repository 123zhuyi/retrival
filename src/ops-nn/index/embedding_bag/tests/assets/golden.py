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

__golden__ = {"kernel": {"embedding_bag": "embedding_bag_golden"}}


def embedding_bag_golden(
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
    Golden function for embedding_bag.
    All the parameters (names and order) follow @embedding_bag_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    dtype_weight = str(weight.dtype)

    if "bfloat16" in dtype_weight:
        weight = torch.from_numpy(weight.view(np.int16)).view(torch.bfloat16)
        if isinstance(per_sample_weight, np.ndarray):
            per_sample_weight = torch.from_numpy(per_sample_weight.view(np.int16)).view(
                torch.bfloat16
            )
    else:
        weight = torch.from_numpy(weight)
        if isinstance(per_sample_weight, np.ndarray):
            per_sample_weight = torch.from_numpy(per_sample_weight)
    indices = torch.from_numpy(indices)
    offsets = torch.from_numpy(offsets)

    mode_dict = {"sum": 0, "mean": 1, "max": 2}
    mode_str = mode

    if mode_str != "sum":
        per_sample_weight = None
    sparse = False
    scale_grad_by_freq = False

    if indices.dim() == 2:
        offsets_2d = None
    else:
        offsets_2d = offsets

    y = torch.nn.functional.embedding_bag(
        indices,
        weight,
        offsets=offsets_2d,
        scale_grad_by_freq=scale_grad_by_freq,
        mode=mode_str,
        sparse=sparse,
        per_sample_weights=per_sample_weight,
        include_last_offset=include_last_offset,
        padding_idx=padding_idx,
    )

    mode_int = mode_dict.get(mode_str)
    indices = indices.reshape(-1)
    if per_sample_weight is not None:
        per_sample_weight = per_sample_weight.reshape(-1)
    result = torch.ops.aten.embedding_bag(
        weight=weight,
        indices=indices,
        offsets=offsets,
        scale_grad_by_freq=scale_grad_by_freq,
        mode=mode_int,
        sparse=sparse,
        per_sample_weights=per_sample_weight,
        include_last_offset=include_last_offset,
        padding_idx=padding_idx,
    )
    _, offset2Bag, bag_size, max_indices = result

    if "bfloat16" in dtype_weight:
        out = y.view(torch.int16).numpy().view(dtype_weight)
    else:
        out = y.numpy()

    return out, offset2Bag.numpy(), bag_size.numpy(), max_indices.numpy()
