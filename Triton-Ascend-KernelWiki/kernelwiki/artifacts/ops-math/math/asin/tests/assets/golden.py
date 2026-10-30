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
        "asin": "asin_golden"
    }
}


def asin_golden(x, **kwargs):
    '''
    Kernel golden for asin.
    All the parameters follow @asin_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    x_dtype = x.dtype
    x_tensor = torch.from_numpy(x)

    if x_dtype.name == "bfloat16" or x_dtype.name == "float16":
        x_tensor = x_tensor.to(torch.float32)

    x_tensor = torch.clamp(x_tensor, -1.0, 1.0)
    result = torch.asin(x_tensor)

    return result.numpy().astype(x_dtype, copy=False)
