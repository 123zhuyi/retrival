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
        "atan_grad": "atan_grad_golden"
    }
}


def atan_grad_golden(y, dy, **kwargs):
    '''
    Kernel golden for atan_grad.
    Computes z = dy / (1 + y * y).

    Args:
        y: forward input tensor (numpy.ndarray)
        dy: upstream gradient tensor (numpy.ndarray)

    Returns:
        z (numpy.ndarray) with the same dtype as y.
    '''
    ori_dtype = y.dtype
    if ori_dtype.name in ("float16", "bfloat16"):
        y_cast = y.astype(np.float32)
        dy_cast = dy.astype(np.float32)
        res = dy_cast / (np.float32(1.0) + y_cast * y_cast)
        return res.astype(ori_dtype, copy=False)
    return dy / (1.0 + y * y)
