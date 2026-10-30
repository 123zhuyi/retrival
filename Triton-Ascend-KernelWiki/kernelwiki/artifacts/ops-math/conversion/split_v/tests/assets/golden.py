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
   	    "split_v": "split_v_golden"
  	}
}
  	
def split_v_golden(x, size_splits, split_dim,
                   num_split: int=0,
                   **kwargs):
    '''
    Kernel golden for split_v.
    All the parameters follow @split_v_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
  	    input_formats, output_formats, input_ori_formats, output_ori_formats,
  	    input_dtypes, output_dtypes.
    '''
    split_dim_val = int(split_dim.item()) if isinstance(split_dim, np.ndarray) else int(split_dim)
    size_splits_list = size_splits.tolist() if isinstance(size_splits, np.ndarray) else list(size_splits)
    
    dim_size = x.shape[split_dim_val]
    if min(size_splits_list) == -1 and size_splits_list.count(-1) == 1:
        size_splits_list[size_splits_list.index(-1)] = dim_size - (sum(size_splits_list) + 1)
    
    indices = []
    start = 0
    for i in size_splits_list:
        start += i
        indices.append(start)
    
    return np.split(x, indices[:-1], axis=split_dim_val)
