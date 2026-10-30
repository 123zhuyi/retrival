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
  	    "fused_mul_add_n": "fused_mul_add_n_golden"
  	}
}

def fused_mul_add_n_golden(x1, x2, x3, **kwargs):
    '''
    Kernel golden for fused_mul_add_n.
    All the parameters follow @fused_mul_add_n_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
	    input_formats, output_formats, input_ori_formats, output_ori_formats,
	    input_dtypes, output_dtypes.
    '''
    dtype = x1.dtype
    if dtype.name in ('float16', 'bfloat16'):
        x1 = x1.astype(numpy.float32)
        x2 = x2.astype(numpy.float32)
        x3 = x3.astype(numpy.float32)
    muls = numpy.multiply(x1, x3)
    add = numpy.add(muls, x2)
    return add.astype(dtype, copy=False)
