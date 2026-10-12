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


__golden__ = {"kernel": {"scatter_elements": "scatter_elements_golden"}}


def scatter_elements_golden(
    data, indices, updates, *, axis=0, reduction="none", **kwargs
):
    """
    Golden function for scatter_elements.
    All the parameters (names and order) follow @scatter_elements_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    x_dtype = data.dtype
    if "bfloat16" in str(x_dtype):
        data = data.astype("float32")
        updates = updates.astype("float32")
    tensor_x = torch.from_numpy(data).clone()
    tensor_index = torch.from_numpy(indices).to(torch.int64)
    tensor_src = torch.from_numpy(updates)

    if reduction == "add":
        out = torch.scatter_reduce(
            tensor_x, axis, tensor_index, tensor_src, reduce="sum"
        )
    elif reduction == "mul":
        out = torch.scatter_reduce(
            tensor_x, axis, tensor_index, tensor_src, reduce="prod"
        )
    else:
        out = torch.scatter(tensor_x, axis, tensor_index, tensor_src)
    res = out.numpy()
    if "bfloat16" in str(x_dtype):
        res = res.astype(x_dtype, copy=False)
    return res
