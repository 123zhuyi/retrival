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

__golden__ = {"kernel": {"index_fill": "index_fill_golden"}}


def _numpy_to_torch_tensor(np_array):
    import torch

    if np_array is None:
        return None
    np_dtype = np_array.dtype.name
    if "bfloat16" in np_dtype:
        np_int16 = np_array.view(dtype=np.int16)
        t_int16 = torch.from_numpy(np_int16)
        return t_int16.view(torch.bfloat16)
    elif "float8" in np_dtype:
        if np_dtype not in ("float8_e4m3fn", "float8_e5m2"):
            raise RuntimeError(
                f"Dtype [{np_dtype}] is not supported to convert to torch.Tensor yet."
            )
        if not hasattr(torch, np_dtype):
            raise RuntimeError(
                f"Current pytorch version [{torch.__version__}] is too old. "
                f"{np_dtype} is not supported."
            )
        return torch.from_numpy(np_array.view(dtype=np.uint8)).view(
            getattr(torch, np_dtype)
        )
    else:
        return torch.from_numpy(np_array)


def _torch_to_numpy_tensor(torch_tensor):
    import torch

    if torch_tensor is None:
        return None
    torch_dtype = torch_tensor.dtype
    torch_dtype_str = str(torch_dtype)
    if torch_dtype == torch.bfloat16:
        t_int16 = torch_tensor.view(torch.int16)
        np_int16 = t_int16.numpy()
        from ml_dtypes import bfloat16

        return np_int16.view(dtype=bfloat16)
    elif "float8" in torch_dtype_str:
        np_dtype_name = torch_dtype_str.split(".")[-1]
        if np_dtype_name == "float8_e5m2":
            from ml_dtypes import float8_e5m2

            np_dtype = float8_e5m2
        elif np_dtype_name == "float8_e4m3fn":
            from ml_dtypes import float8_e4m3fn

            np_dtype = float8_e4m3fn
        else:
            raise RuntimeError(f"Unsupported float8 dtype: {np_dtype_name}")
        np_uint8 = torch_tensor.view(torch.uint8).numpy()
        return np_uint8.view(np_dtype)
    else:
        return torch_tensor.numpy()


def index_fill_golden(x, indices, value, *, dim, **kwargs):
    """
    Golden function for index_fill.
    All the parameters (names and order) follow @index_fill_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    tensor_indices = _numpy_to_torch_tensor(indices).clone()
    if tensor_indices.ndim != 0 and tensor_indices.ndim != 1:
        raise RuntimeError("Indices has to be a vector or scalar for `IndexFill` ops")
    if value is None:
        raise RuntimeError("The input of value can't be null for `IndexFill` ops")
    tensor_value = _numpy_to_torch_tensor(value)
    scalar_value = tensor_value.item()

    tensor_x = _numpy_to_torch_tensor(x).clone()
    tensor_indices = _numpy_to_torch_tensor(indices).to(torch.int64)
    output_y = tensor_x.index_fill(dim, tensor_indices, scalar_value)
    return _torch_to_numpy_tensor(output_y)
