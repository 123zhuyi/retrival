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

__golden__ = {"kernel": {"gather_elements": "gather_elements_golden"}}


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


def gather_elements_golden(x, index, *, dim=0, **kwargs):
    """
    Golden function for gather_elements.
    All the parameters (names and order) follow @gather_elements_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    x_data = x
    index_data = index
    assert np.max(index_data) < x_data.shape[dim], "please check your index data !"

    dtypes = {"uint64": "int64", "uint16": "int16", "uint32": "int32"}
    if x_data.dtype.name in dtypes.keys():
        input_x = _numpy_to_torch_tensor(x_data.view(dtypes[x_data.dtype.name]))
    else:
        input_x = _numpy_to_torch_tensor(x_data)

    index_tensor = torch.from_numpy(index_data).to(torch.int64)
    y = torch.gather(input_x, dim=dim, index=index_tensor)

    return _torch_to_numpy_tensor(y).view(x_data.dtype)
