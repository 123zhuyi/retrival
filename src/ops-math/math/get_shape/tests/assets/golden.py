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

__golden__ = {"kernel": {"get_shape": "get_shape_golden"}}

X_DESCRIPTOR_SIZE = 128


def get_shape_golden(*inputs, **kwargs):
    shapes = []
    for inp in inputs:
        if inp.shape == (X_DESCRIPTOR_SIZE,) and np.issubdtype(inp.dtype, np.integer):
            dim_num = int(inp[3])
            for j in range(dim_num):
                shapes.append(int(inp[4 + j]))
        else:
            shapes.extend(inp.shape)
    return np.array(shapes, dtype=np.int32)
