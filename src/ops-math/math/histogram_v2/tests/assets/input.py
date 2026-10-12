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

__input__ = {
    "kernel": {
        "histogram_v2": "histogram_v2_input"
    }
}

def histogram_v2_input(x, min, max, *, bins=100, y_dtype=3, **kwargs):
    # Cast to float32 for comparison to avoid float16 precision issues
    x_f32 = x.astype(np.float32) if x.dtype == np.float16 else x
    min_val = float(min[0])
    max_val = float(max[0])

    if min_val >= max_val:
        min_val = float(np.min(x_f32))
        max_val = float(np.max(x_f32))
        min[0] = min_val
        max[0] = max_val

    if min_val >= max_val:
        min[0] = min_val - 1
        max[0] = max_val + 1

    return (x, min, max)
