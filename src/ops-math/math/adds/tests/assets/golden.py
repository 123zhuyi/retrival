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
from ml_dtypes import bfloat16


__golden__ = {
    "kernel": {
        "adds": "adds_golden"
    }
}


def adds_golden(x, value: float, **kwargs):
    '''
    Kernel golden for adds.
    All the parameters follow @adds_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    dtype = x.dtype
    if str(dtype) in ("bfloat16", "float16"):
        x_tensor = torch.from_numpy(x.astype("float32"))
    else:
        x_tensor = torch.from_numpy(x)

    if str(dtype) == "int64":
        golden = torch.add(x_tensor, value)
        int64_max = torch.iinfo(torch.int64).max
        int64_min = torch.iinfo(torch.int64).min
        max_boundary = torch.tensor(2**63, dtype=torch.float64, device=golden.device)
        min_boundary = torch.tensor(-(2**63), dtype=torch.float64, device=golden.device)
        golden_int = golden.to(torch.int64)
        golden_int = torch.where(golden >= max_boundary, int64_max, golden_int)
        golden_int = torch.where(golden <= min_boundary, int64_min, golden_int)
        return golden_int.numpy()

    if str(dtype) == "int32":
        golden = torch.add(x_tensor, value)
        int32_max = torch.iinfo(torch.int32).max
        int32_min = torch.iinfo(torch.int32).min
        max_boundary = torch.tensor(2**31, dtype=torch.float32, device=golden.device)
        min_boundary = torch.tensor(-(2**31), dtype=torch.float32, device=golden.device)
        golden_int = golden.to(torch.int32)
        golden_int = torch.where(golden >= max_boundary, int32_max, golden_int)
        golden_int = torch.where(golden <= min_boundary, int32_min, golden_int)
        return golden_int.numpy()

    return torch.add(x_tensor, value).numpy().astype(dtype, copy=False)
