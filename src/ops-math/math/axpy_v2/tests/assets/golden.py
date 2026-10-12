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
  	    "axpy_v2": "axpy_v2_golden"
  	}
}
 	 
 	 
def axpy_v2_golden(x1, x2, alpha, 
                   **kwargs):
    '''
    Kernel golden for axpy_v2.
    All the parameters follow @axpy_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
  	    input_formats, output_formats, input_ori_formats, output_ori_formats,
  	    input_dtypes, output_dtypes.
    '''      
    dtype = x1.dtype
    if dtype.name in ("float16", "bfloat16", "float32"):
        cast_type = np.float32 if dtype.name in ("float16", "bfloat16") else np.float64
        x1_cast = x1.astype(cast_type)
        x2_cast = x2.astype(cast_type)
        data_mul = np.multiply(alpha, x2_cast)
        res = np.add(data_mul, x1_cast).astype(dtype)
    else:
        data_mul = np.multiply(alpha, x2)
        res = np.add(data_mul, x1)
    return res
    