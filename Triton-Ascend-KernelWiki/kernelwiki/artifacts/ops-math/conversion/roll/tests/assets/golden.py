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
import torch


__golden__ = {
    "kernel": {
        "roll": "roll_golden"
    }
}


def roll_golden(x, shifts, dims=None, **kwargs):
    '''
    Kernel golden for roll.
    All the parameters follow @roll_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    dtype_view_mapping = {
        'uint64': 'int64',
        'uint32': 'int32',
        'uint16': 'int16',
        'uint8': 'int8'
    }
    
    if x.dtype.name in dtype_view_mapping:
        x_signed_view = x.view(dtype_view_mapping[x.dtype.name])
        x_torch = torch.from_numpy(x_signed_view)
        rolled_torch = torch.roll(x_torch, shifts=shifts, dims=dims)
        rolled_numpy = rolled_torch.numpy()
        y = rolled_numpy.view(x.dtype)
    else:
        x_torch = torch.from_numpy(x)
        rolled_torch = torch.roll(x_torch, shifts=shifts, dims=dims)
        y = rolled_torch.numpy()
    
    return y