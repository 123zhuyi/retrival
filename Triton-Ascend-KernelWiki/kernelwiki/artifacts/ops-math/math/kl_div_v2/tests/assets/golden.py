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
        "kl_div_v2": "kl_div_v2_golden"
    }
}


def kl_div_v2_golden(x, target, reduction: str = "mean", log_target: bool = False, **kwargs):
    ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
    x_dtype = x.dtype
    
    if "bfloat16" in str(ori_dtype).lower() or "float16" in str(ori_dtype).lower():
        x_tensor = torch.from_numpy(x.astype(np.float32))
        target_tensor = torch.from_numpy(target.astype(np.float32))
        output = torch.nn.functional.kl_div(x_tensor, target_tensor, reduction=reduction, log_target=log_target)
        return output.numpy().astype(x_dtype, copy=False)
    else:
        x_tensor = torch.from_numpy(x)
        target_tensor = torch.from_numpy(target)
        output = torch.nn.functional.kl_div(x_tensor, target_tensor, reduction=reduction, log_target=log_target)
        return output.numpy()