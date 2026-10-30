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
import torch

__golden__ = {
  	"kernel": {
  	    "addcdiv": "addcdiv_golden"
  	}
}

def addcdiv_golden(input_data, x1, x2, value,
                   **kwargs):
    '''
    Kernel golden for addcdiv.
    All the parameters follow @addcdiv_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    data_type = input_data.dtype
    input_data = torch.from_numpy(input_data.astype(numpy.float32))
    x1 = torch.from_numpy(x1.astype(numpy.float32))
    x2 = torch.from_numpy(x2.astype(numpy.float32))
    value = value.item() 
    
    res = torch.addcdiv(input_data, x1, x2, value=value)
    res_np = res.numpy()
    res_np = res_np.astype(data_type, copy=False)

    return res_np
