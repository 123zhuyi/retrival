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

__input__ = {"kernel": {"grouped_dynamic_mx_quant": "grouped_dynamic_mx_quant_input"}}


def grouped_dynamic_mx_quant_input(x, group_index, **kwargs):
    """
    Input function for grouped_dynamic_mx_quant.
    All the parameters (names and order) follow @grouped_dynamic_mx_quant_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Returns:
        List of input tensors (length must match Input count in _def.cpp)
    """
    group_index.sort()
    dim_s = x.shape[0]
    group_index = np.minimum(np.maximum(group_index, 0), dim_s)
    group_index[-1] = dim_s

    return [x, group_index]
