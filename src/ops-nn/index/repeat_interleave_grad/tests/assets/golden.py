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
import torch

__golden__ = {"kernel": {"repeat_interleave_grad": "repeat_interleave_grad_golden"}}


def repeat_interleave_grad_golden(y_grad, repeats, *, axis=-1, **kwargs):
    """
    Golden function for repeat_interleave_grad.
    All the parameters (names and order) follow @repeat_interleave_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    in_dtype = y_grad.dtype
    x_grad_shape = kwargs["output_ori_shapes"][0]
    x = np.random.uniform(-100, 100, x_grad_shape).astype(in_dtype)
    x = torch.from_numpy(x.astype(np.float32))
    y_grad = torch.from_numpy(y_grad.astype(np.float32))
    x.requires_grad_(True)
    repeats = torch.from_numpy(np.array(repeats))
    y = torch.repeat_interleave(x, repeats, axis)
    y.backward(y_grad)
    x_grad = x.grad.detach().numpy().astype(in_dtype)
    return x_grad
