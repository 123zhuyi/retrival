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
        "reduce_std_v2": "reduce_std_v2_golden"
    }
}


def _eliminate_duplicate_axes(axis, input_tensor):
    axis = tuple(set([_ax if _ax >= 0 else len(input_tensor.shape) + _ax for _ax in axis]))
    return axis


def reduce_std_v2_golden(x, dim=None, correction: int = 1, keepdim: bool = False, is_mean_out: bool = True, **kwargs):
    '''
    Kernel golden for reduce_std_v2.
    All the parameters follow @reduce_std_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    import torch
    from packaging import version

    x_dtype = x.dtype
    if "bfloat16" in str(x_dtype).lower():
        x_tensor = torch.from_numpy(x.astype(np.float32))
    else:
        x_tensor = torch.from_numpy(x)

    axis = dim
    axis_d = []
    if not axis:
        for i, _ in enumerate(x_tensor.shape):
            axis_d.append(i)
    else:
        axis_d = axis
    axis_d = _eliminate_duplicate_axes(axis_d, x_tensor)

    torch_version = torch.__version__
    if version.parse(torch_version) < version.parse("2.0.0"):
        if correction > 1:
            raise RuntimeError(f"Invalid corrections:{correction} while torch version {torch_version}")
        unbiased = True
        if correction == 0:
            unbiased = False
        if is_mean_out is None or is_mean_out:
            std, mean = torch.std_mean(x_tensor, dim=axis_d, unbiased=unbiased, keepdim=keepdim)
            if "bfloat16" in str(x_dtype).lower():
                return [std.numpy().astype(x_dtype, copy=False), mean.numpy().astype(x_dtype, copy=False)]
            return [std.numpy(), mean.numpy()]
        std = torch.std(x_tensor, dim=axis_d, correction=correction, keepdim=keepdim)
        if "bfloat16" in str(x_dtype).lower():
            return std.numpy().astype(x_dtype, copy=False)
        return std.numpy()
    else:
        if is_mean_out is None or is_mean_out:
            std, mean = torch.std_mean(x_tensor, dim=axis_d, correction=correction, keepdim=keepdim)
            if "bfloat16" in str(x_dtype).lower():
                return [std.numpy().astype(x_dtype, copy=False), mean.numpy().astype(x_dtype, copy=False)]
            return [std.numpy(), mean.numpy()]
        std = torch.std(x_tensor, dim=axis_d, correction=correction, keepdim=keepdim)
        if "bfloat16" in str(x_dtype).lower():
            return std.numpy().astype(x_dtype, copy=False)
        return std.numpy()
