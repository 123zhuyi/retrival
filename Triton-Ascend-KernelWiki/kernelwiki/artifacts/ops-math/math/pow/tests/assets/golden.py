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
  	    "pow": "pow_golden"
  	}
}

def pow_golden(x1, x2, **kwargs):
    '''
    Kernel golden for pow.
    All the parameters follow @pow_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
	    input_formats, output_formats, input_ori_formats, output_ori_formats,
	    input_dtypes, output_dtypes.
    '''
    import torch

    dtype = x1.dtype
    if "bfloat16" in str(dtype):
        x = torch.from_numpy(x1.view(dtype=numpy.int16)).view(torch.bfloat16).float()
        y = torch.from_numpy(x2.view(dtype=numpy.int16)).view(torch.bfloat16).float()
        res = torch.pow(x, y)
        return res.bfloat16().view(torch.int16).numpy().view(dtype=dtype)
    x = torch.from_numpy(x1)
    y = torch.from_numpy(x2)
    return torch.pow(x, y).numpy()
