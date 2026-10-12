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
        "arg_max_v2": "arg_max_v2_golden"
    }
}


def arg_max_v2_golden(x, dimension, *, dtype=None, **kwargs):
    '''
    Kernel golden for arg_max_v2.
    All the parameters follow @arg_max_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
    output_dtype = kwargs.get("output_dtypes", ["int32"])[0]

    if "bfloat16" in str(ori_dtype).lower():
        x = x.astype("float32")

    axis = int(dimension) % len(x.shape)
    indices = np.argmax(x, axis=axis).astype(output_dtype, copy=False)
    return indices
