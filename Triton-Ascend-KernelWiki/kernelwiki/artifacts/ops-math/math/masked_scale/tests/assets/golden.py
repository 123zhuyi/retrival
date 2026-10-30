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
  	    "masked_scale": "masked_scale_golden"
  	}
}
  	
def masked_scale_golden(x, mask, value: float,
                        **kwargs):
    '''
    Kernel golden for masked_scale.
    All the parameters follow @masked_scale_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
  	    input_formats, output_formats, input_ori_formats, output_ori_formats,
  	    input_dtypes, output_dtypes.
    '''
    x_dtype = x.dtype

    if x_dtype.name not in ('float32', 'float64'):
        x = x.astype('float32')
    if mask.dtype.name not in ('float32', 'float64'):
        mask = mask.astype('float32')
    res = x * mask * value

    return res.astype(x_dtype, copy=False)
