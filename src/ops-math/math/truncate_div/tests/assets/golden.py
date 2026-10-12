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
        "truncate_div": "truncate_div_golden"
    }
}


def _broadcast_to_maxshape(shapes):
    """
    produce broadcast shape
    for example:
        input: shape is [[2, 3], [3, 2, 1], [3, 1, 3]]
        output: [1, 2, 3], [3, 2, 1], [3, 1, 3], [3, 2, 3]
    """
    def _max(_shape):
        no_one_shape = [s for s in _shape if s != 1]
        if len(no_one_shape) == 0:
            max_value = 1
        else:
            max_value = no_one_shape[0]
        return max_value
    max_dim_length = max(len(list(shape)) for shape in shapes)
    input_shapes = []
    for shape in shapes:
        input_shapes.append([1 for _ in range(max_dim_length - len(shape))] + list(shape))
    input_shapes = list(map(list, zip(*input_shapes)))
    max_shape = [_max(shape) for shape in input_shapes]
    input_shapes = list(map(list, zip(*input_shapes)))
    return (*input_shapes, max_shape)


def _numpy_bfloat16():
    try:
        from ml_dtypes import bfloat16
    except ModuleNotFoundError:
        try:
            import tensorflow
            bfloat16 = tensorflow.bfloat16.as_numpy_dtype
        except ModuleNotFoundError:
            raise RuntimeError("ml-dtypes or tensorflow is needed to support bfloat16 dtype!!! "
                                "Please install with `pip3 install ml-dtypes` or `pip3 install tensorflow`")
    return bfloat16


def _numpy_to_torch_tensor(np_array):
    import torch
    if np_array is None:
        return None
    np_dtype = np_array.dtype.name
    if "bfloat16" in np_dtype:
        np_int16 = np_array.view(dtype=np.int16)
        t_int16 = torch.from_numpy(np_int16)
        return t_int16.view(torch.bfloat16)
    else:
        return torch.from_numpy(np_array)


def _torch_to_numpy_tensor(torch_tensor):
    import torch
    if torch_tensor is None:
        return None
    if not isinstance(torch_tensor, torch.Tensor):
        raise RuntimeError(f"Only support torch.Tensor. But got {type(torch_tensor)}")
    torch_dtype = torch_tensor.dtype
    if torch_dtype == torch.bfloat16:
        t_int16 = torch_tensor.view(torch.int16)
        np_int16 = t_int16.numpy()
        return np_int16.view(dtype=_numpy_bfloat16())
    else:
        return torch_tensor.numpy()


def truncate_div_golden(x1, x2, **kwargs):
    '''
    Kernel golden for truncate_div.
    All the parameters follow @truncate_div_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    import torch

    type_int = [torch.int8, torch.int32, torch.int64]
    type_uint = [torch.uint8]
    type_float = [torch.float, torch.float16, torch.bfloat16]

    _, _, res_shape = _broadcast_to_maxshape([x1.shape, x2.shape])
    x2_broadcast = np.broadcast_to(x2, res_shape)
    zero_x2_broadcast_idx = np.where(x2_broadcast == 0)

    output_dtype = kwargs.get("output_dtypes", [None])[0]
    if output_dtype is None:
        output_dtype = str(x1.dtype)
    need_zero_handling = output_dtype in ["int32", "int8", "uint8", "int64"]

    zero_idx = np.where(x2 == 0)
    if len(zero_idx[0]) > 0 and need_zero_handling:
        x2[zero_idx] = 1

    x1_t = _numpy_to_torch_tensor(x1)
    x2_t = _numpy_to_torch_tensor(x2)

    res = 0

    dtype = torch.promote_types(x1_t.dtype, x2_t.dtype)
    if dtype in (torch.int32, torch.int64):
        info = torch.iinfo(dtype)
        min_val, max_val = info.min, info.max
        dangerous_mask = (x1_t == min_val) & (x2_t == -1)
        safe_x2 = torch.where(dangerous_mask, torch.ones_like(x2_t), x2_t)
        res = torch.div(x1_t, safe_x2, rounding_mode="trunc")
    elif dtype == torch.int16:
        dangerous_mask = (x2_t == 0)
        safe_x1 = torch.where(dangerous_mask, torch.full_like(x1_t, -1), x1_t)
        safe_x2 = torch.where(dangerous_mask, torch.full_like(x2_t, -1), x2_t)
        res = torch.div(safe_x1, safe_x2, rounding_mode="trunc")
    else:
        res = torch.div(x1_t, x2_t, rounding_mode='trunc')

    if len(zero_idx[0]) > 0 and need_zero_handling:
        x2[zero_idx] = 0
        if res.dtype in type_int:
            res[zero_x2_broadcast_idx] = -1
        if res.dtype in type_uint:
            res[zero_x2_broadcast_idx] = 255

    if dtype in (torch.int16, torch.int32):
        info = torch.iinfo(dtype)
        min_val, max_val = info.min, info.max

        mask = (x1_t == max_val) & (x2_t == min_val)
        res = torch.where(mask, torch.tensor(0, dtype=dtype), res)

        mask = (x1_t == max_val) & (x2_t == -1)
        res = torch.where(mask, torch.tensor(-max_val, dtype=dtype), res)

        mask = (x1_t == min_val) & (x2_t == -1)
        res = torch.where(mask, torch.tensor(min_val, dtype=dtype), res)

        mask = (x2_t == 0)
        res = torch.where(mask, torch.tensor(-1, dtype=dtype), res)

    if dtype == torch.int64:
        info = torch.iinfo(dtype)
        min_val, max_val = info.min, info.max

        mask = (x1_t == max_val) & (x2_t == 0)
        res = torch.where(mask, torch.tensor(-1, dtype=dtype), res)

        mask = (x1_t == max_val) & (x2_t == -1)
        res = torch.where(mask, torch.tensor(-max_val, dtype=dtype), res)

        mask = (x1_t == min_val) & (x2_t == 0)
        res = torch.where(mask, torch.tensor(-1, dtype=dtype), res)

        mask = (x1_t == min_val) & (x2_t == -1)
        res = torch.where(mask, torch.tensor(min_val, dtype=dtype), res)

        mask = (x1_t >= 0) & (x1_t != max_val) & (x2_t == 0)
        res = torch.where(mask, torch.tensor(4294967295, dtype=dtype), res)

        mask = (x1_t == -1) & (x2_t == 0)
        res = torch.where(mask, torch.tensor(-1, dtype=dtype), res)

    res_np = _torch_to_numpy_tensor(res)

    return res_np.astype(output_dtype, copy=False)
