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
from copy import deepcopy


__golden__ = {"kernel": {"adaptive_avg_pool3d_grad": "adaptive_avg_pool3d_grad_golden"}}


def adaptive_avg_pool3d_grad_golden(y_grad, x, data_format="NDHWC", **kwargs):
    """
    Kernel golden for adaptive_avg_pool3d_grad.
    All the parameters follow @adaptive_avg_pool3d_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch
    import torch.nn.functional as F

    y_grad = deepcopy(y_grad)
    x = deepcopy(x)

    input_formats = kwargs.get("input_formats", [])
    y_grad_format = input_formats[0] if len(input_formats) > 0 else None
    x_format = input_formats[1] if len(input_formats) > 1 else None

    out_size = kwargs.get("output_size", None)
    if out_size is None:
        out_size = kwargs.get("out_size", None)
    if out_size is None:
        out_size = kwargs.get("osize", None)

    # ---- normalize to NCDHW for torch ----
    if x_format == "NDHWC":
        x = x.transpose(0, 4, 1, 2, 3)
    if y_grad_format == "NDHWC":
        y_grad = y_grad.transpose(0, 4, 1, 2, 3)

    if data_format == "NDHWC" and x.ndim == 4:
        x = x.transpose(3, 0, 1, 2)  # -> CDHW
    if data_format == "NDHWC" and y_grad.ndim == 4:
        y_grad = y_grad.transpose(3, 0, 1, 2)

    # infer out_size from y_grad (NCDHW / CDHW)
    if out_size is None:
        if y_grad.ndim == 5:
            out_size = [
                int(y_grad.shape[2]),
                int(y_grad.shape[3]),
                int(y_grad.shape[4]),
            ]
        elif y_grad.ndim == 4:
            out_size = [
                int(y_grad.shape[1]),
                int(y_grad.shape[2]),
                int(y_grad.shape[3]),
            ]
        else:
            raise RuntimeError(f"Unsupported y_grad ndim={y_grad.ndim}")
    else:
        # normalize list[int]
        if isinstance(out_size, (tuple, list)):
            out_size = [int(v) for v in out_size]
        else:
            # e.g. string " [2,3,4] " or scalar not expected; do best effort
            out_size = [int(v) for v in out_size]

    # ---- dtype policy: fp16/bf16 use fp32 compute then cast back ----
    x_dtype = x.dtype

    if str(x_dtype) in ("bfloat16", "float16"):
        x_t = torch.from_numpy(x.astype(np.float32)).requires_grad_(True)
        y_grad_t = torch.from_numpy(y_grad.astype(np.float32))
    else:
        x_t = torch.from_numpy(x).requires_grad_(True)
        y_grad_t = torch.from_numpy(y_grad)

    # ---- forward then backward with provided grad ----
    y = F.adaptive_avg_pool3d(x_t, out_size)
    # y_grad_t shape should match y
    y.backward(y_grad_t)

    grad_x = x_t.grad.detach().cpu().numpy()

    # cast back to original x dtype
    if str(x_dtype) in ("bfloat16", "float16"):
        grad_x = grad_x.astype(x_dtype)
    else:
        grad_x = grad_x.astype(x_dtype, copy=False)

    # ---- convert back to requested layout ----
    if x_format == "NDHWC":
        grad_x = grad_x.transpose(0, 2, 3, 4, 1)
    if data_format == "NDHWC" and grad_x.ndim == 4:
        grad_x = grad_x.transpose(1, 2, 3, 0)

    return grad_x
