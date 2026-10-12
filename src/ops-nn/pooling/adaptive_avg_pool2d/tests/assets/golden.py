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


__golden__ = {"kernel": {"adaptive_avg_pool2d": "adaptive_avg_pool2d_golden"}}


def adaptive_avg_pool2d_golden(x, output_size, **kwargs):
    """
    Kernel golden for adaptive_avg_pool2d.
    All the parameters follow @adaptive_avg_pool2d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import torch
    import torch.nn as nn

    xDtype = x.dtype
    if len(output_size) == 2:
        output_size = tuple(output_size)
    elif len(output_size) == 1:
        output_size = output_size[0]
    elif len(output_size) == 0:
        output_size = (None, None)
    else:
        raise Exception("adaptive_avg_pool2d output_size dim only supports 0,1,2")

    if xDtype in ["float16", "bfloat16"]:
        x = x.astype(np.float32)

    m = nn.AdaptiveAvgPool2d(output_size)
    x_tensor = torch.from_numpy(x)
    output = m(x_tensor)

    output = output.numpy().astype(xDtype)

    return output
