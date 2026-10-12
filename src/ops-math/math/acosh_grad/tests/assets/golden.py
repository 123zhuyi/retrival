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
"""
AcoshGrad kernel-direct golden.

Uses torch.acosh + autograd backward as the golden reference.
Formula: z = dy / sqrt(y^2 - 1)
  y  : original input of acosh (domain >= 1)
  dy : upstream gradient
  z  : gradient w.r.t. the original input

The kernel test context supplies NumPy arrays. Inputs are promoted to float32
for the complete forward/backward calculation, then the result is converted
back to the original output dtype.
"""

from types import SimpleNamespace

import torch
from ttk.utilities.dtypes import numpy_to_torch_tensor, torch_to_numpy_tensor


__golden__ = {
    "kernel": {
        "acosh_grad": "acosh_grad_golden",
    }
}


def acosh_grad_golden(y, dy, **kwargs):
    del kwargs
    context = SimpleNamespace(input_arrays=(y, dy))
    return _acosh_grad(context)


def _acosh_grad(context):
    arrs = context.input_arrays
    y = arrs[0]
    dy = arrs[1]
    # Preserve the original output dtype, including bfloat16 via a lossless bit
    # view, but perform the complete calculation in float32.
    y_t = numpy_to_torch_tensor(y)
    dy_t = numpy_to_torch_tensor(dy)
    out_dtype = y_t.dtype
    y_f = y_t.to(torch.float32).clone().requires_grad_(True)
    dy_f = dy_t.to(torch.float32)

    # Compute acosh forward, then backward to get its input gradient.
    result = torch.autograd.grad(torch.acosh(y_f), y_f, grad_outputs=dy_f)[0]
    return torch_to_numpy_tensor(result.to(out_dtype).detach().cpu())
