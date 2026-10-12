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

__golden__ = {"kernel": {"mish_grad": "mish_grad_golden"}}


def mish_grad_golden(grad, x, tanhx=None, **kwargs):
    """
    Golden function for mish_grad.
    All the parameters (names and order) follow @mish_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    x_dtype = x.dtype
    if x_dtype.name in ("bfloat16", "float16"):
        grad = grad.astype(np.float32)
        x = x.astype(np.float32)

    grad_torch = torch.from_numpy(grad)
    input_x = torch.from_numpy(x)

    if tanhx is None:
        output = torch.ops.aten.mish_backward(grad_torch, input_x)
    else:
        input_y = torch.from_numpy(tanhx.astype(np.float32))
        output = grad_torch * (
            input_y + input_x / (1 + torch.exp(-input_x)) * (1 - input_y * input_y)
        )

    return output.numpy().astype(x_dtype, copy=False)
