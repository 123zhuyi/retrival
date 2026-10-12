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
        "masked_fill": "masked_fill_golden"
    }
}


def masked_fill_golden(x, mask, value, **kwargs):
    '''
    Kernel golden for masked_fill.
    All the parameters follow @masked_fill_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    dtype = x.dtype
    if "bfloat16" in str(dtype):
        x = x.astype("float32")
        value = value.astype("float32")
    
    x_tensor = torch.from_numpy(x)
    mask_tensor = torch.from_numpy(mask)
    value_scalar = value[0]
    res = x_tensor.masked_fill(mask_tensor, value_scalar).numpy()
    
    if "bfloat16" in str(dtype):
        res = res.astype(dtype, copy=False)
    return res