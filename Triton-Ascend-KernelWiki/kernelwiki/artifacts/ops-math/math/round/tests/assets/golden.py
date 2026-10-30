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
        "round": "round_golden"
    }
}


def round_golden(x, decimals: int = 0, **kwargs):
    '''
    Kernel golden for round.
    All the parameters follow @round_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    import torch
    
    ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
    x_dtype = x.dtype
    
    if "int" in str(x_dtype):
        return np.round(x, decimals)
    
    if ori_dtype and "float16" in str(ori_dtype).lower():
        x_float = x.astype(np.float32)
        res = torch.round(torch.from_numpy(x_float), decimals=decimals).numpy()
        return res.astype(x_dtype, copy=False)
    elif ori_dtype and "bfloat16" in str(ori_dtype).lower():
        x_float = x.astype(np.float32)
        res = torch.round(torch.from_numpy(x_float), decimals=decimals).numpy()
        return res.astype(x_dtype, copy=False)
    else:
        return torch.round(torch.from_numpy(x), decimals=decimals).numpy()