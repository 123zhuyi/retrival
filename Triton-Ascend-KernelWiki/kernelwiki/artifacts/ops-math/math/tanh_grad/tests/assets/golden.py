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
  	    "tanh_grad": "tanh_grad_golden"
  	}
}

def tanh_grad_golden(y, dy, **kwargs):
    '''
    Kernel golden for tanh_grad.
    All the parameters follow @tanh_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
	    input_formats, output_formats, input_ori_formats, output_ori_formats,
	    input_dtypes, output_dtypes.
    '''
    dtype = y.dtype
    is_mix_dtype = y.dtype != dy.dtype
    y = y.astype("float32")
    dy = dy.astype("float32")
    data_square = numpy.multiply(y, y)
    data_mul = numpy.multiply(data_square, -1)
    data_add = numpy.add(data_mul, 1)
    result = numpy.multiply(data_add, dy)

    if is_mix_dtype:
        return result

    return result.astype(dtype, copy=False)
