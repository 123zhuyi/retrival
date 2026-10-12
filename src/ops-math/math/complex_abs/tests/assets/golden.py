#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

__golden__ = {"kernel": {"complex_abs": "complex_abs_golden"}}


def complex_abs_golden(x, **kwargs):
    """
    Kernel golden for complex_abs.
    Computes y = |x| for complex input.

    Args:
        x: input tensor (numpy.ndarray)
            - complex64: complex64 type, shape same as output
            - complex32: complex32 type, shape same as output

    kwargs may contain: input_dtypes, output_dtypes, short_soc_version, etc.

    Returns:
        y (numpy.ndarray): absolute value of each complex element.
            - complex64 -> float32, shape same as input
            - complex32 -> float16, shape same as input
    """
    ori_dtype = kwargs.get("input_dtypes", ["complex64"])[0]

    if ori_dtype and "complex32" in str(ori_dtype).lower():
        x_tensor = torch.from_numpy(x)
        complex_tensor = torch.view_as_complex(x_tensor.to(torch.float16))
        res_tensor = torch.abs(complex_tensor)
        return res_tensor.numpy()
    elif ori_dtype and "complex64" in str(ori_dtype).lower():
        return np.abs(x)
    else:
        return np.abs(x)
