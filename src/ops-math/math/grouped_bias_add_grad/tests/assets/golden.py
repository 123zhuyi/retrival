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


__golden__ = {
    "kernel": {
        "grouped_bias_add_grad": "grouped_bias_add_grad_golden"
    }
}


def grouped_bias_add_grad_golden(grad_y, group_idx=None, group_idx_type: int = 0, **kwargs):
    if group_idx is not None:
        group_idx_arr = np.array(group_idx.tolist() if hasattr(group_idx, 'tolist') else list(group_idx), dtype=np.int64)
        
        if group_idx_type == 1:
            end_indices = np.cumsum(group_idx_arr)
        else:
            end_indices = group_idx_arr
        
        G = len(end_indices)
        H = grad_y.shape[-1]
        grad_bias = np.zeros((G, H), dtype=np.float64)
        
        start = 0
        for j in range(G):
            end = int(end_indices[j])
            if end > start:
                grad_bias[j] = np.sum(grad_y[start:end].astype(np.float64), axis=0)
            start = end
        
        return grad_bias
    else:
        return np.sum(grad_y.astype(np.float64), axis=1)