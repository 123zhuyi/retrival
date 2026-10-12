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

__golden__ = {
    "kernel": {
        "mul_no_nan": "mul_no_nan_golden"
    }
}


def mul_no_nan_golden(x1, x2, **kwargs):
    '''
    Kernel golden for mul_no_nan.
    All the parameters follow @mul_no_nan_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.

    Semantics: y = (x2 == 0) ? 0 : x1 * x2 (element-wise, NumPy broadcasting).
    The mask is computed BEFORE forming the product, so for inputs that would
    produce 0*inf=NaN or 0*NaN=NaN, the result is forced to 0. This is the
    core differentiator vs plain Mul.
    '''
    dtype = x1.dtype
    if dtype == np.int32:
        # Integer path: no NaN/Inf, kernel computes in native int32.
        prod = (x1 * x2).astype(np.int32)
        out = np.where(x2 == 0, np.int32(0), prod).astype(np.int32)
    else:
        # Float / half / bf16 path: lift to fp32 for the intermediate compute
        # to match the kernel's MulNoNanFloatCast template for fp16/bf16, and
        # to be a no-op for the native fp32 template. Cast back at the end.
        x1f = x1.astype(np.float32)
        x2f = x2.astype(np.float32)
        mask = (x2f == np.float32(0.0))
        prod = x1f * x2f
        # np.where picks element-wise, so wherever mask is True we get 0 even
        # if prod[i] is NaN/Inf -- this is exactly the MulNoNan semantics.
        out = np.where(mask, np.float32(0.0), prod).astype(dtype)
    return out
