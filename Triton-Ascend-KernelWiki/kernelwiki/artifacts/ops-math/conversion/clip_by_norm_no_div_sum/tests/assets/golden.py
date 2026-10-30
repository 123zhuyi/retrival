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
        "clip_by_norm_no_div_sum": "clip_by_norm_no_div_sum_golden"
    }
}


def clip_by_norm_no_div_sum_golden(x, greater_zeros, select_ones, maximum_ones, **kwargs):
    '''
    Golden function for clip_by_norm_no_div_sum.
    All the parameters (names and order) follow @clip_by_norm_no_div_sum_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Formula:
        inner_sel = (x > greater_zeros) ? x : select_ones
        sqrt_val  = sqrt(inner_sel)
        sel_out   = (x <= greater_zeros) ? x : sqrt_val
        y         = max(sel_out, maximum_ones)

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    if "bfloat16" in str(x.dtype):
        x = x.astype("float32")
        greater_zeros = greater_zeros.astype("float32")
        select_ones = select_ones.astype("float32")
        maximum_ones = maximum_ones.astype("float32")

    inner_sel = np.where(x > greater_zeros, x, select_ones)
    sqrt_val = np.sqrt(inner_sel)
    sel_out = np.where(x <= greater_zeros, x, sqrt_val)
    res = np.maximum(sel_out, maximum_ones)

    if "bfloat16" in str(x.dtype):
        return res.astype(x.dtype, copy=False)
    return res
