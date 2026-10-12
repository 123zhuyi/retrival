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

__input__ = {"kernel": {"dynamic_quant_v2": "dynamic_quant_v2_input"}}


def dynamic_quant_v2_input(x, smooth_scales=None, group_index=None, **kwargs):
    """
    Input function for dynamic_quant_v2.
    All the parameters (names and order) follow @dynamic_quant_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Returns:
        List of input tensors (length must match Input count in _def.cpp)
    """
    if group_index is not None:
        S = np.prod(x.shape[:-1])
        E = group_index.shape[0]

        group_index = np.random.choice(np.arange(1, S + 1), size=E, replace=False)
        group_index.sort()
        group_index[-1] = S
        group_index = group_index.astype("int32")

    return [x, smooth_scales, group_index]
