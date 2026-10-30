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
        "dot": "dot_golden"
    }
}

def dot_golden(input_x, input_y, **kwargs):
    '''
    Kernel golden for dot.
    All the parameters follow @dot_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    ori_dtype = input_x.dtype
    
    if ori_dtype.name in ("float16", "bfloat16"):
        input_x_cast = input_x.astype(np.float32)
        input_y_cast = input_y.astype(np.float32)
        res = np.dot(input_x_cast, input_y_cast)
        return res.astype(ori_dtype, copy=False)
    elif ori_dtype.name in ("int8", "uint8"):
        input_x_cast = input_x.astype(np.int32)
        input_y_cast = input_y.astype(np.int32)
        res = np.dot(input_x_cast, input_y_cast)
        return res.astype(np.int32, copy=False)
    else:
        res = np.dot(input_x, input_y)
        return res