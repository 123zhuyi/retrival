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
import torch

__golden__ = {
    "kernel": {
        "range": "_range_golden"
    },
    "aclnn": {
        "aclnnArange": "aclnn_arange_golden",
        "aclnnRange": "aclnn_range_golden"
    }
}

def _bfloat16_conversion(dtypes):
    result = []
    for dt in dtypes:
        if 'bfloat16' in str(dt):
            result.append(np.dtype('bfloat16'))
        else:
            result.append(dt)
    return result

def _torch_dtype_conversion(dtypes):
    mapping = {
        'float16': torch.float16,
        'float32': torch.float,
        'float': torch.float,
        'int32': torch.int32,
        'int64': torch.int64,
        'bfloat16': torch.bfloat16,
        'double': torch.double,
    }
    result = []
    for dt in dtypes:
        dt_str = str(dt) if hasattr(dt, '__str__') else dt
        for key in mapping:
            if key in dt_str.lower():
                result.append(mapping[key])
                break
        else:
            result.append(torch.float)
    return result

def _range_golden(start, limit, delta, *, is_closed=False, **kwargs):
    '''
    Golden function for range kernel.
    All the parameters (names and order) follow range_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor with range values
    '''
    output_dtypes = kwargs.get('output_dtypes', ['float32'])
    input_dtypes = kwargs.get('input_dtypes', ['float32'])
    
    src_type = _bfloat16_conversion(output_dtypes)[0]
    torch_dtype = _torch_dtype_conversion(output_dtypes)[0]
    input_dtypes_torch = _torch_dtype_conversion(input_dtypes)
    
    start_val = torch.tensor(start, dtype=input_dtypes_torch[0]).item()
    stop_val = torch.tensor(limit, dtype=input_dtypes_torch[1]).item()
    step_val = torch.tensor(delta, dtype=input_dtypes_torch[2]).item()

    if step_val == 0 or (step_val > 0 and start_val > stop_val) or (step_val < 0 and start_val < stop_val):
        return np.array([], dtype=src_type)

    if "bfloat16" in str(src_type):
        if is_closed:
            golden = torch.range(start_val, stop_val, step_val, dtype=torch_dtype).float().numpy()
        else:
            golden = torch.arange(start_val, stop_val, step_val, dtype=torch_dtype).float().numpy()
    else:
        if is_closed:
            golden = torch.range(start_val, stop_val, step_val, dtype=torch_dtype).numpy()
        else:
            golden = torch.arange(start_val, stop_val, step_val, dtype=torch_dtype).numpy()
    
    return golden.astype(src_type)

def aclnn_arange_golden(start, end, step, out, **kwargs):
    '''
    Aclnn golden for aclnnArange.
    All the parameters (name & order) follow \
        function `aclnnArangeGetWorkspaceSize` in @aclnn_arange.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    '''
    return torch.arange(start, end, step, dtype=out.dtype)

def aclnn_range_golden(start, limit, delta, is_closed, out, **kwargs):
    '''
    Aclnn golden for aclnnRange.
    All the parameters (name & order) follow \
        function `aclnnRangeGetWorkspaceSize` in @aclnn_range.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    '''
    if is_closed:
        return torch.range(start, limit, delta, dtype=out.dtype)
    else:
        return torch.arange(start, limit, delta, dtype=out.dtype)