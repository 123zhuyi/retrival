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
        "tensor_equal": "tensor_equal_golden"
    }
}


def tensor_equal_golden(input_x, input_y, **kwargs):
    '''
    Kernel golden for tensor_equal.
    All the parameters follow @tensor_equal_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    import torch
    
    x_dtype = input_x.dtype
    y_dtype = input_y.dtype
    
    if str(x_dtype) == "bfloat16" and str(y_dtype) == "bfloat16":
        input_x = input_x.astype(np.float32)
        input_y = input_y.astype(np.float32)
    
    tensor_x = torch.tensor(input_x)
    tensor_y = torch.tensor(input_y)
    result = torch.equal(tensor_x, tensor_y)
    
    return np.array(result)