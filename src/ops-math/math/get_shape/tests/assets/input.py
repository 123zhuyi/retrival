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

__input__ = {"kernel": {"get_shape": "get_shape_input"}}

X_DESCRIPTOR_SIZE = 128


def _build_shape_descriptor(ori_shape):
    desc = np.zeros(X_DESCRIPTOR_SIZE, dtype=np.int64)
    desc[3] = np.int64(len(ori_shape))
    for i, dim in enumerate(ori_shape):
        desc[4 + i] = np.int64(dim)
    return desc


def get_shape_input(*input_arrays, **kwargs):
    desc_arrays = []
    for arr in input_arrays:
        ori_shape = arr.shape
        desc = _build_shape_descriptor(ori_shape)
        desc_arrays.append(desc)

    return list(desc_arrays)
