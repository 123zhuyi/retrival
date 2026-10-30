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
import torch

__golden__ = {
  	"kernel": {
  	    "data_compare": "data_compare_golden"
  	}
}


def data_compare_golden(x1, x2, *, atol: float = 0.0, rtol: float = 0.0,
                        **kwargs):
    '''
    Kernel golden for data_compare.

    Math:
        a = x1.astype(float32)
        b = x2.astype(float32)
        mismatch = (|a - b| > atol + rtol * |b|).astype(float32)
        output = sum(mismatch)

    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    if x1.size == 0:
        return np.array(0.0, dtype=np.float32)

    if "bfloat16" in x1.dtype.name:
        t1 = torch.from_numpy(x1.view(np.int16)).view(torch.bfloat16).float()
    else:
        t1 = torch.from_numpy(np.asarray(x1)).float()

    if "bfloat16" in x2.dtype.name:
        t2 = torch.from_numpy(x2.view(np.int16)).view(torch.bfloat16).float()
    else:
        t2 = torch.from_numpy(np.asarray(x2)).float()

    mismatch = (torch.abs(t1 - t2) > atol + rtol * torch.abs(t2)).to(torch.float32)
    result = torch.sum(mismatch)

    return result.numpy()
