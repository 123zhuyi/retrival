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
  	    "split": "split_golden"
  	}
}
  	
def split_golden(split_dim, x,
                 num_split: int,
                 **kwargs):
    '''
    Kernel golden for split.
    All the parameters follow @split_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
  	    input_formats, output_formats, input_ori_formats, output_ori_formats,
  	    input_dtypes, output_dtypes.
    '''
    split_dim_val = int(split_dim.item()) if isinstance(split_dim, np.ndarray) else int(split_dim)

    input_ori_formats = kwargs.get('input_ori_formats', ['ND'])
    x_ori_format = input_ori_formats[0] if input_ori_formats else 'ND'
    if x_ori_format == "NHWC" and split_dim_val == 3:
        split_dim_val = 1

    return np.split(x, num_split, axis=split_dim_val)
