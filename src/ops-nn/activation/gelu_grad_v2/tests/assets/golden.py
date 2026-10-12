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

__golden__ = {"kernel": {"gelu_grad_v2": "gelu_grad_v2_golden"}}


def _cdf(input_x):
    """
    alpha = 1 / sqrt(2)
    return erf(alpha * x) *0.5 + 0.5
    """
    from scipy import special

    bbb = 1 / np.sqrt(2)
    erf_input = np.multiply(input_x, bbb)
    erf_res = special.erf(erf_input)
    mul_res = np.multiply(erf_res, 0.5)
    res = np.add(mul_res, 0.5)
    return res


def _pdf(input_x):
    """
    beta = 1 / sqrt(2 * pi)
    return beta * exp(-0.5*t*t)
    """
    aaa = 1 / np.sqrt(2 * np.pi)
    mul_res = np.multiply(input_x, input_x)
    mul_res_1 = np.multiply(mul_res, -0.5)
    exp_res = np.exp(mul_res_1)
    res = np.multiply(exp_res, aaa)
    return res


def gelu_grad_v2_golden(dy, x, *, approximate="none", **kwargs):
    """
    Golden function for gelu_grad_v2.
    All the parameters (names and order) follow @gelu_grad_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    input_dtype = dy.dtype

    has_improve_precision = False
    if input_dtype != "float32":
        dy = dy.astype(np.float32)
        x = x.astype(np.float32)
        has_improve_precision = True

    cdf = _cdf(x)
    pdf = _pdf(x)
    res_1 = np.multiply(x, pdf)
    result5 = np.add(res_1, cdf)

    result = np.multiply(dy, result5)
    if has_improve_precision:
        result = result.astype(input_dtype)
    return result
