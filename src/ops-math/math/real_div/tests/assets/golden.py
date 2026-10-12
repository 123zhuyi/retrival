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
  	    "real_div": "real_div_golden"
  	}
}

def real_div_golden(x1, x2, **kwargs):
    '''
    Kernel golden for real_div.
    All the parameters follow @real_div_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
	    input_formats, output_formats, input_ori_formats, output_ori_formats,
	    input_dtypes, output_dtypes.
    '''
    ori_dtype = x1.dtype
    if "bfloat16" in str(ori_dtype):
        x1, x2 = x1.astype("float32"), x2.astype("float32")

    res = numpy.true_divide(x1, x2)
    output_dtype = kwargs.get("output_dtypes", [res.dtype])[0]
    if "bfloat16" in str(output_dtype):
        from ml_dtypes import bfloat16
        return res.astype(bfloat16, copy=False)
    return res.astype(output_dtype, copy=False)
