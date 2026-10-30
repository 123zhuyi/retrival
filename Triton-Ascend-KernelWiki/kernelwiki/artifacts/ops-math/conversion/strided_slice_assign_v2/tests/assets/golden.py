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
        "strided_slice_assign_v2": "strided_slice_assign_v2_golden"
    }
}


def strided_slice_assign_v2_golden(var, input_value, begin, end, strides, axes=None, **kwargs):
    '''
    Kernel golden for strided_slice_assign_v2.
    All the parameters follow @strided_slice_assign_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    
    var:          要修改的张量 (numpy.ndarray)
    input_value:  要赋的值 (numpy.ndarray)
    begin:        切片起始位置 (numpy.ndarray of int64)
    end:          切片结束位置 (numpy.ndarray of int64)
    strides:      切片步长 (numpy.ndarray of int64)
    axes:         需要切片的轴 (numpy.ndarray of int64, 可选)
    返回:         修改后的 var (numpy.ndarray)
    '''
    begin = np.asarray(begin).flatten().tolist()
    end = np.asarray(end).flatten().tolist()
    strides = np.asarray(strides).flatten().tolist()
    
    input_dtype = var.dtype
    if str(input_dtype) == "bfloat16":
        var = var.astype(np.float32)
        input_value = input_value.astype(np.float32)
    
    ndim = var.ndim
    
    if axes is not None:
        axes = np.asarray(axes).flatten().tolist()
    else:
        axes = list(range(len(begin)))
    
    slices = [slice(None)] * ndim
    for i, axis in enumerate(axes):
        if axis < 0:
            axis += ndim
        slices[axis] = slice(int(begin[i]), int(end[i]), int(strides[i]))
    
    var[tuple(slices)] = input_value
    
    if str(input_dtype) == "bfloat16":
        var = var.astype(input_dtype)
    
    return var
