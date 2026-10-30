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
from typing import Union
from functools import reduce as func_reduce

__golden__ = {
  	"kernel": {
  	    "as_strided": "as_strided_golden"
  	}
}
 	 
dtype_width_map = {
    "complex32": 4,
    "complex64": 8,
    "double": 8,
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "uint64": 8,
    "int32": 4,
    "uint32": 4,
    "int16": 2,
    "uint16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
    "float8_e5m2": 1,
    "float8_e4m3fn": 1,
    "hifloat8": 1
}

def as_strided_golden(x, size, stride, storage_offset=None,
                      **kwargs):
    '''
    Kernel golden for as_strided.
    All the parameters follow @as_strided_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
  	    input_formats, output_formats, input_ori_formats, output_ori_formats,
  	    input_dtypes, output_dtypes.
  	'''
    dtype_name = x.dtype.name
    if dtype_name not in dtype_width_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    storage_offset = storage_offset[0] if storage_offset else 0
    ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
    
    if ori_dtype == 'complex32':
        import torch
        t_x = torch.from_numpy(x).view(torch.complex32)
        output = torch.as_strided(t_x, size.tolist(), stride.tolist(), storage_offset)
        output = torch.view_as_real(output).numpy()
    else:
        output = numpy.lib.stride_tricks.as_strided(x[storage_offset:], size, stride * dtype_width_map.get(dtype_name, 1))
    return output


    
