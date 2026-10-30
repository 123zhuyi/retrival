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
 	 
__golden__ = {
  	"kernel": {
  	    "add_n": "add_n_golden"
  	}
}
 	 
def add_n_golden(x,
                 N: int=1,
                 **kwargs):
    '''
    Kernel golden for add_n.
    All the parameters follow @add_n_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
 	    input_formats, output_formats, input_ori_formats, output_ori_formats,
 	    input_dtypes, output_dtypes.
 	'''      
    def my_sum(tensors):
        result = tensors[0]
        for i in range(len(tensors) - 1):
            result = result + tensors[i + 1]
        return result

    def group_add(tensors, group_size=8):
        N = len(tensors)
        if N == 1:
            return tensors[0]
        u = N % group_size
        u = u + group_size if u <= 1 else u

        result = my_sum(tensors[:u])
        while u < N:
            result = result + my_sum(tensors[u:u + group_size])
            u = u + group_size
        return result
    
    # same as tf.add_n in tf2.x
    out = group_add(x)

    return out