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
   	    "floor_mod": "floor_mod_golden"
   	}
}

def numpy_to_torch_tensor(x_np):
    """
    Convert numpy array to torch tensor, handling bfloat16 specifically.
    bfloat16 in numpy is stored as int16; view back to torch.bfloat16 then float.
    """
    if "bfloat16" in str(x_np.dtype):
        return torch.from_numpy(x_np.view(dtype=np.int16)).view(torch.bfloat16).float()
    return torch.from_numpy(x_np)

def floor_mod_golden(x1, x2,
                     **kwargs):
    '''
    Kernel golden for floor_mod.
    All the parameters follow @floor_mod_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
  	    input_formats, output_formats, input_ori_formats, output_ori_formats,
  	    input_dtypes, output_dtypes.
    '''
    x1_dtype = x1.dtype
    type_int = [torch.int64, torch.int32]
    type_float = [torch.float, torch.float16, torch.bfloat16]

    # 除零保护：在副本上操作，避免污染入参
    res_shape = np.broadcast_shapes(x1.shape, x2.shape)
    X2_broadcast = np.broadcast_to(x2, res_shape)
    zero_X2_broadcast_idx = np.where(X2_broadcast == 0)

    zero_idx = np.where(x2 == 0)
    has_zero = zero_idx[0].size > 0
    x2_safe = x2.copy()
    if has_zero:
        x2_safe[zero_idx] = 1

    x1_t = numpy_to_torch_tensor(x1)
    x2_t = numpy_to_torch_tensor(x2_safe)
    res = torch.remainder(x1_t, x2_t)

    # 除零保护
    if has_zero:
        if res.dtype in type_int:
            res[zero_X2_broadcast_idx] = -1
        if res.dtype in type_float:
            res[zero_X2_broadcast_idx] = torch.nan

    if "bfloat16" in str(x1_dtype):
        return res.bfloat16().view(torch.int16).numpy().view(dtype=x1_dtype)
    return res.numpy()
