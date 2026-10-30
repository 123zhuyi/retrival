#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
        "add_mat_mat_elements": "add_mat_mat_elements_golden"
    }
}


def add_mat_mat_elements_golden(c, a, b, beta, alpha, **kwargs):
    '''
    Kernel golden for add_mat_mat_elements.
    Computes c_out = c * beta + alpha * a * b (elementwise, with broadcasting).

    Args:
        c: tensor (numpy.ndarray)
        a: tensor (numpy.ndarray)
        b: tensor (numpy.ndarray)
        beta: scalar tensor (1-element numpy.ndarray)
        alpha: scalar tensor (1-element numpy.ndarray)

    Returns:
        c_out (numpy.ndarray) with the same dtype as c.
    '''
    ori_dtype = c.dtype
    if ori_dtype.name in ("float16", "bfloat16"):
        cast_type = np.float32
        c_cast = c.astype(cast_type)
        a_cast = a.astype(cast_type)
        b_cast = b.astype(cast_type)
        beta_cast = beta.astype(cast_type)
        alpha_cast = alpha.astype(cast_type)
        res = c_cast * beta_cast + alpha_cast * a_cast * b_cast
        return res.astype(ori_dtype, copy=False)
    return c * beta + alpha * a * b
