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
  	    "logical_or": "logical_or_golden"
  	}
}
  	
def logical_or_golden(x1, x2,
                    **kwargs):
    '''
    Kernel golden for logical_or.
    All the parameters follow @logical_or_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
  	    input_formats, output_formats, input_ori_formats, output_ori_formats,
  	    input_dtypes, output_dtypes.
    '''
    shape_list = np.broadcast_shapes(x1.shape, x2.shape)
    x1 = x1.astype("float16")
    x2 = x2.astype("float16")
    x1 = np.broadcast_to(x1, shape_list)
    x2 = np.broadcast_to(x2, shape_list)
    return np.maximum(x1, x2).astype("int8")
