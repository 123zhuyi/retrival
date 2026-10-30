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
  	    "lerp": "lerp_golden"
  	}
}
  	
def lerp_golden(start, end, weight,
                **kwargs):
    '''
    Kernel golden for lerp.
    All the parameters follow @lerp_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
  	    input_formats, output_formats, input_ori_formats, output_ori_formats,
  	    input_dtypes, output_dtypes.
    '''
    
    dtype = start.dtype
    
    if "bfloat16" in str(start.dtype):
        start = start.astype("float32")
    if "bfloat16" in str(end.dtype):
        end = end.astype("float32")
    if "bfloat16" in str(weight.dtype):
        weight = weight.astype("float32")

    start_tensor = torch.from_numpy(start).to(torch.float32)
    end_tensor = torch.from_numpy(end).to(torch.float32)
    weight_tensor = torch.from_numpy(weight).to(torch.float32)
    
    golden = torch.lerp(start_tensor, end_tensor, weight_tensor).numpy()
    return golden.astype(dtype, copy=False)
