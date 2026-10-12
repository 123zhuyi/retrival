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

__golden__ = {"kernel": {"abs_grad": "abs_grad_golden"}}


def abs_grad_golden(y, dy, **kwargs):
    """
    Kernel golden for abs_grad.
    Computes z = dy * sign(y).

    Args:
        y: forward input tensor (numpy.ndarray)
        dy: upstream gradient tensor (numpy.ndarray)

    Returns:
        z (numpy.ndarray) with the same dtype as y.
    """
    return dy * np.sign(y)
