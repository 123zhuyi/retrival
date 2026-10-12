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

__golden__ = {"kernel": {"gelu_v2": "gelu_v2_golden"}}


def gelu_v2_golden(x, *, approximate="none", **kwargs):
    """
    Golden function for gelu_v2.
    All the parameters (names and order) follow @gelu_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    from scipy import special

    input_dtype = x.dtype
    if input_dtype != "float32":
        x = x.astype("float32")

    if approximate == "tanh":
        x_cubed = np.power(x, 3)
        inner_term = x + 0.044715 * x_cubed
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        scaled_term = sqrt_2_over_pi * inner_term
        tanh_term = np.tanh(scaled_term)
        tanh_plus_1 = 1 + tanh_term
        res = 0.5 * x * tanh_plus_1
    else:
        erf_input = x / np.sqrt(2)
        res0 = special.erf(erf_input)
        res1 = (res0 / 2) + 0.5
        res = np.multiply(res1, x)

    if input_dtype != res.dtype:
        res = res.astype(input_dtype, copy=False)
    return res
