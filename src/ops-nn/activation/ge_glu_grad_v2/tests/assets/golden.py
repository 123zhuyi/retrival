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

__golden__ = {"kernel": {"ge_glu_grad_v2": "ge_glu_grad_v2_golden"}}


COEFFICIENT_A1 = -0.0713548162726002527220
COEFFICIENT_A2 = -1.5957691216057308
COEFFICIENT_A3 = -0.21406444881780074632901625683959062
COEFFICIENT_A4 = -1.5957691216057307117597842397375274738

TH_MAX = 3.92
TH_MIN = -3.92
COEFFICIENT_1 = 0.70710678118
COEFFICIENT_2 = 0.3989422804
COEFFICIENT_3 = 0.53443748819e-1
COEFFICIENT_4 = 0.75517016694e1
COEFFICIENT_5 = 0.10162808918e3
COEFFICIENT_6 = 0.13938061484e4
COEFFICIENT_7 = 0.50637915060e4
COEFFICIENT_8 = 0.29639384698e5
COEFFICIENT_9 = 0.31212858877e2
COEFFICIENT_10 = 0.39856963806e3
COEFFICIENT_11 = 0.30231248150e4
COEFFICIENT_12 = 0.13243365831e5
COEFFICIENT_13 = 0.26267224157e5


def erf_compute_with_simplified_formula(input_x):
    import torch

    torch_input_x = torch.from_numpy(input_x)
    torch_erf = torch.erf(torch_input_x)
    res = torch_erf.numpy()
    return res


def cdf(x):
    """
    alpha = 1 / sqrt(2)
    return erf(alpha * x) *0.5 + 0.5
    """
    erf_input = x * COEFFICIENT_1
    erf_res = erf_compute_with_simplified_formula(erf_input)
    res = erf_res * 0.5 + 0.5
    return res


def pdf(x):
    """
    beta = 1 / sqrt(2 * pi)
    return beta * exp(-0.5*t*t)
    """
    exp_res = np.exp(-0.5 * x * x) * COEFFICIENT_2
    return exp_res


def gelu_grad_tanh(dy, x):
    """
    g1(x): 1.0 / (exp(x*(x^2*a1+a2))+1)
    g2(x): x^2*a3 + a4
    f(x): (x*(g1-1)*g2+1)*g1*dy
    """
    g1 = 1.0 / (np.exp(x * (x * x * COEFFICIENT_A1 + COEFFICIENT_A2)) + 1)
    g2 = x * x * COEFFICIENT_A3 + COEFFICIENT_A4
    dx = (x * (g1 - 1) * g2 + 1) * g1 * dy
    return dx


def gelu_grad_erf(dy, x):
    """
    result = cdf(x) + x * pdf(x)
    """
    cdf_res = cdf(x)
    pdf_res = pdf(x)
    dx = (cdf_res + x * pdf_res) * dy
    return dx


def ge_glu_grad_v2_golden_v2(dy, x, gelu, dim, approximate, activate_left):
    dy_dtype = dy.dtype
    if dy_dtype.name == "bfloat16":
        dy = dy.astype(np.float32)
        x = x.astype(np.float32)
        gelu = gelu.astype(np.float32)
    x1, x2 = np.split(x, 2, axis=dim)
    if activate_left:
        x1, x2 = x2, x1

    x1 = dy * x1
    if dy_dtype == np.float16:
        x1 = x1.astype(np.float32)
        x2 = x2.astype(np.float32)

    if approximate == 1:
        dx2 = gelu_grad_tanh(x1, x2)
    else:
        dx2 = gelu_grad_erf(x1, x2)

    if dy_dtype == np.float16:
        dx2 = dx2.astype(dy_dtype)

    dx1 = dy * gelu
    if activate_left:
        dx1, dx2 = dx2, dx1

    x_grad = np.concatenate((dx1, dx2), axis=dim)
    if dy_dtype.name == "bfloat16":
        x_grad = x_grad.astype(dy_dtype)
    return x_grad


def ge_glu_grad_v2_golden(
    dy, x, gelu, *, dim=-1, approximate=1, activate_left=False, **kwargs
):
    """
    Golden function for ge_glu_grad_v2.
    All the parameters (names and order) follow @ge_glu_grad_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    x_grad = ge_glu_grad_v2_golden_v2(dy, x, gelu, dim, approximate, activate_left)
    return [x_grad]
