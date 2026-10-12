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
        "masked_select_v3": "masked_select_v3_golden"
    }
}


def _numpy_to_torch_tensor(arr):
    """将 numpy 数组转换为 torch tensor"""
    return torch.from_numpy(arr)


def masked_select_v3_golden(x, mask, **kwargs):
    '''
    Kernel golden for masked_select_v3.
    All the parameters follow @masked_select_v3_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    dtypes = {'uint64': 'int64', 'uint16': 'int16', 'uint32': 'int32', 'bfloat16': 'int16'}
    
    if x.dtype.name in dtypes.keys():
        input_x = _numpy_to_torch_tensor(x.view(dtypes[x.dtype.name]))
    else:
        input_x = _numpy_to_torch_tensor(x)
    
    mask_tensor = _numpy_to_torch_tensor(mask)
    y = torch.masked_select(input_x, mask_tensor).numpy()
    
    if x.dtype.name in dtypes.keys():
        y = y.view(x.dtype)
    return y