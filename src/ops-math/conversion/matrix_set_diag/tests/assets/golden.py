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
        "matrix_set_diag": "matrix_set_diag_golden"
    }
}


def matrix_set_diag_golden(x, diagonal, **kwargs):
    '''
    Kernel golden for matrix_set_diag.
    All the parameters follow @matrix_set_diag_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    y = x.copy()
    
    if x.ndim == 2:
        n = min(x.shape[0], x.shape[1])
        diag_len = diagonal.shape[0] if diagonal.ndim == 1 else diagonal.shape[-1]
        length = min(n, diag_len)
        for i in range(length):
            y[i, i] = diagonal[i] if diagonal.ndim == 1 else diagonal[..., i]
    elif x.ndim == 3:
        batch_size = x.shape[0]
        n = min(x.shape[1], x.shape[2])
        for b in range(batch_size):
            diag_len = diagonal[b].shape[0] if diagonal.ndim == 2 else diagonal.shape[-1]
            length = min(n, diag_len)
            for i in range(length):
                y[b, i, i] = diagonal[b, i] if diagonal.ndim == 2 else diagonal[..., i]
    else:
        n = min(x.shape[-2], x.shape[-1])
        diag_len = diagonal.shape[-1]
        length = min(n, diag_len)
        for i in range(length):
            y[..., i, i] = diagonal[..., i]
    
    return y