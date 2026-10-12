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
import numpy

__golden__ = {
  	"kernel": {
  	    "select": "select_golden"
  	}
}

def select_golden(condition, x1, x2, **kwargs):
    '''
    Kernel golden for select.
    All the parameters follow @select_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
	    input_formats, output_formats, input_ori_formats, output_ori_formats,
	    input_dtypes, output_dtypes.
    '''
    import torch

    conditionDim = condition.ndim
    conditionShape = condition.shape
    x1Shape = x1.shape
    extraDims = len(x1Shape) - len(conditionShape)

    if extraDims >= 0:
        newShape = conditionShape + (1,) * extraDims
        conditionExpanded = condition.reshape(newShape)
    else:
        conditionExpanded = condition

    dtype = x1.dtype
    con = torch.from_numpy(conditionExpanded)
    if "bfloat16" in str(dtype):
        x = torch.from_numpy(x1.view(dtype=numpy.int16)).view(torch.bfloat16).float()
        y = torch.from_numpy(x2.view(dtype=numpy.int16)).view(torch.bfloat16).float()
        res = torch.where(con, x, y)
        return res.float().numpy().astype(dtype, copy=False)
    x = torch.from_numpy(x1)
    y = torch.from_numpy(x2)
    return torch.where(con, x, y).numpy()
