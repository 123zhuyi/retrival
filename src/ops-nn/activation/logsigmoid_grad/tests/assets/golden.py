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


__golden__ = {"kernel": {"log_sigmoid_grad": "log_sigmoid_grad_golden"}}


def log_sigmoid_grad_golden(grads, features, **kwargs):
    """
    Golden function for logsigmoid_grad.
    All the parameters (names and order) follow @logsigmoid_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch
    import torch.nn.functional as F

    dtype = features.dtype
    if "float16" in str(dtype) or "bfloat16" in str(dtype):
        grads = grads.astype("float32")
        features = features.astype("float32")

    grad_output = torch.from_numpy(grads)
    input_tensor = torch.from_numpy(features)

    if not input_tensor.requires_grad:
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        grad_output = grad_output.clone().detach().requires_grad_(True)

    log_sigmoid = F.logsigmoid(input_tensor)
    log_sigmoid.sum().backward()
    auto_grad = input_tensor.grad * grad_output

    auto_grad = auto_grad.detach().numpy()
    if "float16" in str(dtype) or "bfloat16" in str(dtype):
        auto_grad = auto_grad.astype(dtype)
    return auto_grad
