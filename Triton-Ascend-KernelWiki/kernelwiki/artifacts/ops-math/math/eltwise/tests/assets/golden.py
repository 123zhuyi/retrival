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
import functools

__golden__ = {
    "kernel": {
        "eltwise": "eltwise_golden"
    }
}

def eltwise_golden(x, mode: int = 1, coeff = [], **kwargs):
    '''
    Kernel golden for eltwise.
    All the parameters follow @eltwise_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    
    Args:
        x: list of input tensors (1~32 tensors), each is numpy.ndarray
        mode: computation mode (0=PRODUCT, 1=SUM, 2=MAX), default 1
        coeff: list of coefficients for SUM mode, default []
    '''
    if not isinstance(x, (list, tuple)):
        inputs = [x]
    else:
        inputs = list(x)
    
    input_num = len(inputs)
    if input_num == 0:
        return np.array([])
    
    input_shape = inputs[0].shape
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x * y, input_shape)
    
    for i in range(input_num):
        inputs[i] = np.reshape(inputs[i], fuseshape)
    
    input0 = inputs[0]
    if mode == 1 and len(coeff) == input_num and coeff[0] != 1:
        input0 = np.multiply(input0, coeff[0])
    
    res = None
    if input_num == 1:
        input0 = np.add(input0, 0)
        res = input0
    elif input_num > 1:
        for i in range(1, input_num):
            inputn = inputs[i]
            if mode == 0:
                input0 = np.multiply(input0, inputn)
            elif mode == 2:
                input0 = np.maximum(input0, inputn)
            else:
                if not coeff:
                    input0 = np.add(input0, inputn)
                elif coeff[i] == 1:
                    input0 = np.add(input0, inputn)
                else:
                    inputn = np.multiply(inputn, coeff[i])
                    input0 = np.add(input0, inputn)
        res = input0
    
    output_dtypes = kwargs.get("output_dtypes", [])
    if output_dtypes:
        return res.astype(output_dtypes[0])
    else:
        return res.astype(inputs[0].dtype)