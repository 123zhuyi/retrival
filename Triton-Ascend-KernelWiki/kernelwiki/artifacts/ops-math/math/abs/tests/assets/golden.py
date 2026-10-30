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
  	    "abs": "abs_golden"
  	}
}
 	 
def abs_golden(x,
               **kwargs):
    '''
    Kernel golden for abs.
    All the parameters follow @abs_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
 	    input_formats, output_formats, input_ori_formats, output_ori_formats,
 	    input_dtypes, output_dtypes.
 	'''
    ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
    input_x_dtype = x.dtype
    
    if ori_dtype and "complex32" in str(ori_dtype).lower():    
        real, imag = np.split(x, 2, axis=-1)
        res = np.sqrt(real * real + imag * imag)
        return res
    elif ori_dtype and "bfloat16" in str(ori_dtype).lower():  
        x = np.float32(x)
        return np.abs(x).astype(input_x_dtype, copy=False)
    else:
        return np.abs(x)

    