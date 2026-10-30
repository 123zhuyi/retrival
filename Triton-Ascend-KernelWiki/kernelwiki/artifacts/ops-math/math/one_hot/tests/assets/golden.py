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
  	    "one_hot": "one_hot_golden"
  	}
}
  	
def one_hot_golden(x, depth, on_value, off_value,
                   axis: int=-1,
                   **kwargs):
    '''
    Kernel golden for one_hot.
    All the parameters follow @one_hot_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes, 
  	    input_formats, output_formats, input_ori_formats, output_ori_formats,
  	    input_dtypes, output_dtypes.
    '''
    import tensorflow.compat.v1 as tf
    from tensorflow.python.ops import gen_array_ops
    tf.disable_eager_execution()

    data_dtype = on_value.dtype
    on_value_const = tf.constant(on_value, shape=(), dtype=data_dtype)
    off_value_const = tf.constant(off_value, shape=(), dtype=data_dtype)
    axis = max(axis, -1)
    out = gen_array_ops.one_hot(x, max(int(depth), 0), on_value_const, off_value_const, axis)
    with tf.Session() as sess:
        res = sess.run(out)
    return res
