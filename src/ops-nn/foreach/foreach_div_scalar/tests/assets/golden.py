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

__golden__ = {"kernel": {"foreach_div_scalar": "foreach_div_scalar_golden"}}


def foreach_div_scalar_golden(x, scalar, **kwargs):
    """
    Golden function for foreach_div_scalar.
    All the parameters (names and order) follow @foreach_div_scalar_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Tuple of output tensors
    """
    import torch

    dtype = x[0].dtype
    dtype_str = str(dtype)
    scalar_val = scalar[0] if isinstance(scalar, np.ndarray) else scalar

    input_pts: list = []
    for inp in x:
        if "float16" in dtype_str:
            inp = inp.astype(np.float32)
        inp_pt = torch.from_numpy(inp)
        input_pts.append(inp_pt)

    input_tuple = tuple(input_pt for input_pt in input_pts)
    result_pts = torch._foreach_div(input_tuple, scalar_val)

    return tuple(result_pt.numpy().astype(dtype) for result_pt in result_pts)
