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
        "drop_out_do_mask": "drop_out_do_mask_golden"
    }
}


def revert_bit(n):
    result = 0
    for i in range(8):
        result <<= 1
        result |= n & 1
        n >>= 1
    return result


def revert_array_bit(arr):
    res = []
    for item in arr.flatten():
        res.append(revert_bit(item))
    return np.array(res, dtype=np.uint8).reshape(arr.shape)


def drop_out_do_mask_golden(x, mask, keep_prob, **kwargs):
    '''
    Golden function for drop_out_do_mask.
    All the parameters (names and order) follow @drop_out_do_mask_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    short_soc_version = kwargs.get('short_soc_version', '')
    dtype = str(x.dtype)

    if short_soc_version in ("Ascend950",):
        if dtype in ("bfloat16", "float16"):
            x = x.astype("float32")
            keep_prob = keep_prob.astype("float32")
        if keep_prob.flat[0] == 1.0:
            x = x.astype(dtype)
            return x
        elif keep_prob.flat[0] == 0.0:
            y_out = np.zeros(x.shape, dtype=dtype)
            return y_out
    else:
        if str(x.dtype) == "bfloat16":
            x = x.astype("float32")
            keep_prob = keep_prob.astype("float32")

    shape_x = x.shape
    x_scale = x * (1.0 / keep_prob)
    mask = revert_array_bit(mask)
    mask_dtype = np.unpackbits(mask, axis=-1).astype(x.dtype)

    size_x = 1
    x_scale = x_scale.flatten()
    for i in shape_x:
        size_x = size_x * i
    expect = x_scale
    mask_dtype = mask_dtype[:size_x]
    expect[mask_dtype == 0] = 0
    output = expect.reshape(shape_x)
    return output.astype(dtype)
