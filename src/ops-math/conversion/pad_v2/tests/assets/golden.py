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
from collections import deque


__golden__ = {
    "kernel": {
        "pad_v2": "pad_v2_golden"
    }
}


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


def _numpy_float8_e5m2():
    from ml_dtypes import float8_e5m2
    return float8_e5m2


def _numpy_float8_e4m3fn():
    from ml_dtypes import float8_e4m3fn
    return float8_e4m3fn


def _numpy_float8_e8m0():
    from en_dtypes import float8_e8m0
    return float8_e8m0


def _numpy_float4_e2m1():
    from en_dtypes import float4_e2m1
    return float4_e2m1


def _numpy_float4_e1m2():
    from en_dtypes import float4_e1m2
    return float4_e1m2


def _numpy_hifloat8():
    from en_dtypes import hifloat8
    return hifloat8


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


def pad_v2_golden(x, paddings, constant_values, **kwargs):
    '''
    Kernel golden for pad_v2.
    All the parameters follow @pad_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    import torch

    paddings_arr = np.array(paddings).astype(np.int64)
    if paddings_arr.ndim == 1:
        paddings_arr = paddings_arr.reshape(-1, 2)
    paddings_val = paddings_arr.tolist()
    constant_values_arr = constant_values

    input_formats = kwargs.get('input_formats', ())
    x_format = input_formats[0] if input_formats and len(input_formats) > 0 else 'ND'

    pad_shape = deque()
    for i in range(len(paddings_val)):
        pad_shape.append(paddings_val[len(paddings_val) - 1 - i][0])
        pad_shape.append(paddings_val[len(paddings_val) - 1 - i][1])

    if x_format == "NC1HWC0":
        pad_shape.appendleft(0)
        pad_shape.appendleft(0)

    dtypes = {'uint8': 'int8', 'uint16': 'int16', 'uint32': 'int32', 'uint64': 'int64'}
    fp8dtypes = ["hifloat8", "float8_e8m0", "float8_e4m3fn", "float8_e5m2"]
    fp4dtypes = ["float4_e2m1", "float4_e1m2"]

    if x.dtype.name in dtypes.keys():
        x_tensor = _numpy_to_torch_tensor(x.view(dtypes[x.dtype.name]))
        cv = constant_values_arr[0]
    elif x.dtype.name == "bfloat16":
        x_tensor = _numpy_to_torch_tensor(x).to(torch.float32)
        cv = float(constant_values_arr[0])
    elif x.dtype.name in fp8dtypes:
        x_tensor = _numpy_to_torch_tensor(x.view(np.int8))
        constant_values_arr = constant_values_arr.view(np.int8)
        cv = constant_values_arr.item() if constant_values_arr.size == 1 else constant_values_arr
    elif x.dtype.name in fp4dtypes:
        x_tensor = _numpy_to_torch_tensor(x.astype(np.float32))
        constant_values_arr = constant_values_arr.astype(np.float32)
        cv = constant_values_arr.item() if constant_values_arr.size == 1 else constant_values_arr
    else:
        x_tensor = _numpy_to_torch_tensor(x)
        cv = constant_values_arr[0]

    golden = torch.constant_pad_nd(x_tensor, tuple(pad_shape), cv)

    if x.dtype.name in dtypes.keys():
        golden = golden.numpy().view(x.dtype)
    elif x.dtype.name == "bfloat16":
        golden = _torch_to_numpy_tensor(golden.to(torch.bfloat16))
    elif x.dtype.name in fp8dtypes:
        np_dtype = eval(f"_numpy_{x.dtype.name.split('.')[-1]}()")
        golden = golden.numpy().view(np_dtype)
    elif x.dtype.name in fp4dtypes:
        np_dtype = eval(f"_numpy_{x.dtype.name.split('.')[-1]}()")
        golden = golden.numpy().astype(np_dtype)
    else:
        golden = golden.numpy()

    return golden
