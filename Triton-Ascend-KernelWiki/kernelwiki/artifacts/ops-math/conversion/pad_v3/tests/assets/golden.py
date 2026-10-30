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
        "pad_v3": "pad_v3_golden"
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


def _pad_and_slice(arr, pad_width, mode='edge'):
    pad_width = tuple(pad_width)
    if len(pad_width) != arr.ndim:
        raise ValueError(f"pad_width length ({len(pad_width)}) must match array ndim ({arr.ndim})")

    result = arr

    for i, (left, right) in enumerate(pad_width):
        if left == 0 and right == 0:
            continue

        if left > 0 or right > 0:
            pad_tuple = [(0, 0)] * arr.ndim
            lv = left if left > 0 else 0
            rv = right if right > 0 else 0
            pad_tuple[i] = (lv, rv)
            result = np.pad(result, pad_tuple, mode=mode)

        if left < 0 or right < 0:
            crop_left = abs(left) if left < 0 else 0
            crop_right = abs(right) if right < 0 else 0
            slice_start = crop_left
            slice_end = -crop_right if crop_right > 0 else None
            slices = [slice(None)] * arr.ndim
            slices[i] = slice(slice_start, slice_end)
            result = result[tuple(slices)]

    return result


def pad_v3_golden(x, paddings, constant_values=None, mode="constant", paddings_contiguous=True, **kwargs):
    '''
    Kernel golden for pad_v3.
    All the parameters follow @pad_v3_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    import torch

    input_formats = kwargs.get('input_formats', ())
    x_format = input_formats[0] if input_formats and len(input_formats) > 0 else 'ND'

    paddings = np.array(paddings).astype(np.int64)
    if paddings.ndim == 1:
        if paddings_contiguous:
            paddings = paddings.reshape(-1, 2)
        else:
            paddings = paddings.reshape(2, -1)

    if mode == 'constant':
        pad_shape = deque()
        if paddings_contiguous == True:
            for i in range(len(paddings)):
                pad_shape.append(paddings[len(paddings) - 1 - i][0])
                pad_shape.append(paddings[len(paddings) - 1 - i][1])
        else:
            for i in range(len(paddings[0])):
                pad_shape.append(paddings[0][len(paddings[0]) - 1 - i])
                pad_shape.append(paddings[1][len(paddings[1]) - 1 - i])

        if x_format == "NC1HWC0":
            pad_shape.appendleft(0)
            pad_shape.appendleft(0)

        dtypes = {'uint8': 'int8', 'uint16': 'int16', 'uint32': 'int32', 'uint64': 'int64'}

        if x.dtype.name in dtypes.keys():
            x_tensor = _numpy_to_torch_tensor(x.view(dtypes[x.dtype.name]))
        elif x.dtype.name == "bfloat16":
            x_tensor = _numpy_to_torch_tensor(x).to(torch.float32)
        else:
            x_tensor = _numpy_to_torch_tensor(x)

        if constant_values is not None:
            const_val = constant_values.item() if isinstance(constant_values, np.ndarray) else constant_values
        else:
            const_val = 0
        golden = torch.constant_pad_nd(x_tensor, tuple(pad_shape), const_val)

        if x.dtype.name in dtypes.keys():
            golden = golden.numpy().view(x.dtype)
        elif x.dtype.name == "bfloat16":
            golden = _torch_to_numpy_tensor(golden.to(torch.bfloat16))
        else:
            golden = golden.numpy()
    elif mode == 'reflect' or mode == 'symmetric' or mode == 'edge':
        pad_shape = list()
        if paddings_contiguous:
            pad_shape = paddings
        else:
            pad_shape = np.stack(paddings, axis=1).ravel().tolist()
            pad_shape = np.array(pad_shape).reshape(-1, 2).tolist()

        orig_dtype = x.dtype
        if orig_dtype.name == "bfloat16":
            x = _numpy_to_torch_tensor(x).to(torch.float32).numpy()

        neg_pad = _numpy_to_torch_tensor(np.array(pad_shape))
        neg_mask = neg_pad < 0
        indices = torch.nonzero(neg_mask, as_tuple=False)

        if indices.size(0) == 0:
            golden = np.pad(x, pad_shape, mode=mode)
        else:
            golden = _pad_and_slice(x, pad_shape, mode=mode)

        if orig_dtype.name == "bfloat16":
            golden = _torch_to_numpy_tensor(_numpy_to_torch_tensor(golden).to(torch.bfloat16))
    else:
        raise ValueError(f"Unsupported pad mode: {mode}")

    return golden
