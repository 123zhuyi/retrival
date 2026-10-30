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
        "rsqrt_grad": "rsqrt_grad_golden"
    }
}


def rsqrt_grad_golden(y, dy, **kwargs):
    '''
    Kernel golden for rsqrt_grad.
    All the parameters follow @rsqrt_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    input0, input1 = y, dy
    
    if input0.dtype in ("int32",):
        rsqrt_const = np.array(-2, dtype=input0.dtype)
        res_mul = np.multiply(input1, input0)
        res_mul1 = np.multiply(input0, input0)
        res_mul2 = np.divide(res_mul, rsqrt_const).astype('int32')
        result = np.multiply(res_mul2, res_mul1)
        result = result.astype(input0.dtype)
        return result
    
    if input0.dtype in ("int8",):
        input0 = input0.astype("float32")
        input1 = input1.astype("float32")
        rsqrt_const = np.array(-0.5, dtype='float32')
        res_mul = np.multiply(input1, input0)
        res_mul1 = np.multiply(input0, input0)
        res_mul2 = np.multiply(res_mul, rsqrt_const)
        result = np.multiply(res_mul2, res_mul1).astype('int8')
        return result
    
    rsqrt_const = np.array(-0.5, dtype=input0.dtype)
    res_mul = np.multiply(input1, input0)
    res_mul1 = np.multiply(input0, input0)
    res_mul2 = np.multiply(rsqrt_const, res_mul)
    result = np.multiply(res_mul2, res_mul1)
    
    return result
