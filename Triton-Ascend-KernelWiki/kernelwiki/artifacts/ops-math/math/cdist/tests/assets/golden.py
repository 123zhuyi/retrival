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
        "cdist": "cdist_golden"
    }
}


def cdist_golden(x1, x2, p: float = 2.0, **kwargs):
    ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
    x_dtype = x1.dtype
    
    if "bfloat16" in str(ori_dtype).lower() or "float16" in str(ori_dtype).lower():
        x1_tensor = torch.from_numpy(x1.astype(np.float32))
        x2_tensor = torch.from_numpy(x2.astype(np.float32))
        output = torch.cdist(x1_tensor, x2_tensor, p=p)
        return output.numpy().astype(x_dtype, copy=False)
    else:
        x1_tensor = torch.from_numpy(x1)
        x2_tensor = torch.from_numpy(x2)
        output = torch.cdist(x1_tensor, x2_tensor, p=p)
        return output.numpy()