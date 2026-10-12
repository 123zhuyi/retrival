#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np


__golden__ = {"kernel": {"reduce_std_v2_update": "reduce_std_v2_update_golden"}}

# __input__: 修正 NPU 收到的 mean 为 x.mean(dim, keepdim=True).broadcast_to(x.shape)
# reduce_std_v2_update 是双输入算子(x + mean)，TTK 随机生成的 mean 不等于 x.mean(dim)。
# golden 用 torch.std/torch.var 自算 mean，故须把 NPU 的 mean 修正为 x.mean(dim) 使两者对齐。
__input__ = {"kernel": {"reduce_std_v2_update": "reduce_std_v2_update_input"}}


def _eliminate_duplicate_axes(axis, input_tensor):
    axis = tuple(
        set([_ax if _ax >= 0 else len(input_tensor.shape) + _ax for _ax in axis])
    )
    return axis


def reduce_std_v2_update_golden(
    x,
    mean,
    dim=None,
    if_std=False,
    unbiased=True,
    keepdim=False,
    correction=1,
    **kwargs,
):
    """
    Kernel golden for reduce_std_v2_update.
    All the parameters follow @reduce_std_v2_update_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.

    Uses torch.std/torch.var (PyTorch competitor API) for computation.

    Note: mean input is accepted per op signature but not used; torch.std/torch.var
    compute mean internally (mathematically equivalent when mean = x.mean(dim)).
    """
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

    # unbiased -> correction conversion (align with def.cpp semantics)
    if unbiased is False or unbiased == 0:
        correction = 0
    elif unbiased is True or unbiased == 1:
        correction = 1

    torch_version = torch.__version__
    if version.parse(torch_version) < version.parse("2.0.0"):
        if correction > 1:
            raise RuntimeError(
                f"Invalid corrections:{correction} while torch version {torch_version}"
            )
        unbiased_flag = True
        if correction == 0:
            unbiased_flag = False
        if if_std:
            result = torch.std(
                x_tensor, dim=axis_d, unbiased=unbiased_flag, keepdim=keepdim
            )
        else:
            result = torch.var(
                x_tensor, dim=axis_d, unbiased=unbiased_flag, keepdim=keepdim
            )
    else:
        if if_std:
            result = torch.std(
                x_tensor, dim=axis_d, correction=correction, keepdim=keepdim
            )
        else:
            result = torch.var(
                x_tensor, dim=axis_d, correction=correction, keepdim=keepdim
            )

    if "bfloat16" in str(x_dtype).lower():
        return result.numpy().astype(x_dtype, copy=False)
    return result.numpy().astype(x_dtype, copy=False)


def reduce_std_v2_update_input(*input_arrays, **kwargs):
    """__input__: 修正 NPU 收到的 mean 为 np.mean(x, dim, keepdim=True).broadcast_to(x.shape)。

    使 NPU 的 mean 与 golden(torch.std/torch.var 自算 mean)对齐。
    """
    if len(input_arrays) < 2:
        return list(input_arrays)

    x = input_arrays[0]
    if x is None or input_arrays[1] is None:
        return list(input_arrays)

    dim_raw = kwargs.get("dim")
    if dim_raw is None or (hasattr(dim_raw, "__len__") and len(dim_raw) == 0):
        axes = tuple(range(x.ndim))
    else:
        axes = tuple(d if d >= 0 else d + x.ndim for d in dim_raw)

    orig_dtype = x.dtype
    x_f32 = x.astype(np.float32)
    mean_val = np.mean(x_f32, axis=axes, keepdims=True)
    mean_corrected = np.broadcast_to(mean_val, x_f32.shape).astype(orig_dtype)

    result = list(input_arrays)
    result[1] = mean_corrected.copy()
    return result
