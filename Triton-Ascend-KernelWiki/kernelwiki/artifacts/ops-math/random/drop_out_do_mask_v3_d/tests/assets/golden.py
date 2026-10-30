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
        "drop_out_do_mask_v3_d": "drop_out_do_mask_v3_d_golden"
    }
}


def _get_prob_dtype(prob):
    return getattr(prob, "dtype", np.float32)


def _convert_prob_by_dtype(prob, dtype):
    dtype_str = str(dtype)
    prob_arr = np.asarray(prob)
    if "float16" in dtype_str and "bfloat16" not in dtype_str:
        return prob_arr.astype(np.float16).astype(np.float32)[()]
    return prob_arr.astype(np.float32)[()]


def drop_out_do_mask_v3_d_golden(x, mask, keep_prob, **kwargs):
    '''
    Kernel golden for drop_out_do_mask_v3_d.
    All the parameters follow @drop_out_do_mask_v3_d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    input_x = x
    input_mask = mask

    x_dtype = input_x.dtype
    prob_dtype = _get_prob_dtype(keep_prob)
    keep_prob = _convert_prob_by_dtype(keep_prob, prob_dtype)

    if keep_prob == np.float32(1.0):
        return input_x
    if keep_prob == np.float32(0.0):
        return np.zeros(input_x.shape, dtype=x_dtype)

    input_x_calc = input_x.astype(np.float32)
    prob_reciprocal = np.float32(1.0) / keep_prob

    x_scale = input_x_calc.flatten() * prob_reciprocal
    input_mask = input_mask.flatten()[:input_x.size]
    x_scale[input_mask == 0] = np.float32(0.0)
    output = x_scale.reshape(input_x.shape)
    return output.astype(x_dtype)
