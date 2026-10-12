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


__golden__ = {
    "kernel": {"masked_scatter_with_position": "masked_scatter_with_position_golden"}
}


def masked_scatter_with_position_golden(x, mask, position, updates, **kwargs):
    """
    Golden function for masked_scatter_with_position.
    All the parameters (names and order) follow @masked_scatter_with_position_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    dtype = x.dtype
    if "bfloat16" in str(dtype):
        x = x.view("int16")
        updates = updates.view("int16")

    x = torch.from_numpy(x)
    tmpX = x.clone()
    mask = torch.from_numpy(mask)
    position = torch.from_numpy(position)
    updates = torch.from_numpy(updates)

    res = tmpX.masked_scatter_(mask, updates).numpy()
    if "bfloat16" in str(dtype):
        res = res.view(dtype)
    return res
