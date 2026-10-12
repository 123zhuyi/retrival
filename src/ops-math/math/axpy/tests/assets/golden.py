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
  	    "axpy": "axpy_golden"
  	}
}
 	 
def axpy_golden(x1, x2,
                alpha: float, 
                **kwargs):
    '''
    Kernel golden for axpy.
    All the parameters follow @axpy_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
  	    input_formats, output_formats, input_ori_formats, output_ori_formats,
  	    input_dtypes, output_dtypes.
    '''      
    x1_dtype = x1.dtype  
    if x1_dtype.name in ("float16", "bfloat16", "float32"):
        # axpy uses vaxpy/vmula instructions in 950 for precision
        cast_type = np.float32 if x1_dtype.name in ("float16", "bfloat16") else np.float64
        alpha = np.float32(alpha)
        alpha_cast = alpha.astype(cast_type)
        x1_cast = x1.astype(cast_type)
        x2_cast = x2.astype(cast_type)
        res_muls = np.multiply(x2_cast, alpha_cast)
        res = np.add(x1_cast, res_muls).astype(x1_dtype)
    else:
        # int32
        if alpha != 1:
            # add+muls use fp32
            x1_cast = x1.astype(np.float32)
            x2_cast = x2.astype(np.float32)
            res_muls = np.multiply(x2_cast, alpha)
            res_tmp = np.add(x1_cast, res_muls)
            res = res_tmp.astype(np.int32)
        else:
            # if alpha == 1
            res = np.add(x2, x1)
    return res.astype(x1_dtype, copy=False)