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
  	    "adjacent_difference": "adjacent_difference_golden"
  	}
}
 	 
 	 
def adjacent_difference_golden(x, 
                               y_dtype: int=3, 
                               **kwargs):
    '''
    Kernel golden for adjacent_difference.
    All the parameters follow @adjacent_difference_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
 	    input_formats, output_formats, input_ori_formats, output_ori_formats,
 	    input_dtypes, output_dtypes.
 	'''      
    import torch
    
    x = x.flatten()
    total_size = np.prod(x.shape)
    ret = np.zeros(x.shape, np.int32)

    if total_size == 0:
        return ret

    ret[0] = 0
    ret[1:] = (x[1:] != x[0:-1]).astype(np.int32)

    if (y_dtype == 9):
        ret = ret.astype(np.int64)
    return ret
    