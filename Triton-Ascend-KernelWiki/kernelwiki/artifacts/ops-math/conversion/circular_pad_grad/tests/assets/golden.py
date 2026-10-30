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
        "circular_pad_grad": "circular_pad_grad_golden"
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


def _cal_out_shape(in_shape, paddings):
    out_shape = []
    offset = len(in_shape) - len(paddings)
    for i in range(len(in_shape)):
        if i < len(in_shape) - len(paddings):
            out_shape.append(in_shape[i])
        else:
            out_shape.append(in_shape[i] - paddings[i - offset][0] - paddings[i - offset][1])
    return out_shape


def _torch_direct_invoke_circular(grad_output_np, y_shape_list, pad_temp):
    import torch

    pad = []
    for dim in reversed(pad_temp):
        pad.append(dim[0])
        pad.append(dim[1])

    origin_dtype = grad_output_np.dtype.name
    if grad_output_np.dtype.name == "bfloat16":
        grad_output = _numpy_to_torch_tensor(grad_output_np).to(torch.float32)
    else:
        grad_output = _numpy_to_torch_tensor(grad_output_np)
    x = torch.zeros(y_shape_list, dtype=grad_output.dtype)

    grad_output = grad_output.unsqueeze(0)
    x = x.unsqueeze(0)

    x.requires_grad_(True)
    out = torch.nn.functional.pad(x, pad, "circular")

    loss = (grad_output * out).sum()
    loss.backward()

    golden = x.grad

    golden.squeeze(0)
    if origin_dtype == "bfloat16":
        golden = _torch_to_numpy_tensor(golden.to(torch.bfloat16))
    else:
        golden = golden.numpy()
    return golden


def _numpy_pad_v3_grad_circular(grad_output, input_shape, pad):
    dim = len(input_shape)

    true_dtype = grad_output.dtype
    grad_output = np.array(grad_output, dtype=np.float32)
    grad_input = np.zeros(input_shape, dtype=np.float32)
    for idx_out in np.ndindex(grad_output.shape):
        idx_in = tuple(
            (idx_out[d] - pad[d][0]) % input_shape[d] for d in range(dim)
        )
        grad_input[idx_in] += grad_output[idx_out]
    grad_input = np.array(grad_input, dtype=true_dtype)
    return grad_input


def circular_pad_grad_golden(x, paddings, **kwargs):
    '''
    Kernel golden for circular_pad_grad.
    All the parameters follow @circular_pad_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    grad_output = x

    paddings_arr = np.array(paddings).astype(np.int64)
    if paddings_arr.ndim == 1:
        paddings_arr = paddings_arr.reshape(-1, 2)
    pad_shape = paddings_arr.tolist()

    y_shape = _cal_out_shape(grad_output.shape, pad_shape)

    if grad_output.ndim <= 3:
        grad = _torch_direct_invoke_circular(grad_output, y_shape, pad_shape)
    else:
        grad = _numpy_pad_v3_grad_circular(grad_output, y_shape, pad_shape)
    return grad
