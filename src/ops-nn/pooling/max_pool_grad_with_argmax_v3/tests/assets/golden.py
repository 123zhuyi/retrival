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
from copy import deepcopy

__golden__ = {
    "kernel": {"max_pool_grad_with_argmax_v3": "max_pool_grad_with_argmax_v3_golden"}
}


def max_pool_grad_with_argmax_v3_golden(
    x,
    grad,
    argmax,
    *,
    ksize,
    strides,
    pads,
    dtype=3,
    dilation=[1, 1],
    ceil_mode=False,
    data_format="NCHW",
    **kwargs,
):
    """
    Golden function for max_pool_grad_with_argmax_v3.
    All the parameters (names and order) follow @max_pool_grad_with_argmax_v3_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    input_x = deepcopy(x)
    input_x_format = kwargs["input_formats"][0]
    inpu_x_dtype = input_x.dtype

    attr_re_ksize = ksize
    attr_re_strides = strides
    attr_re_pads = pads
    attr_op_dtype = dtype
    attr_op_dilations = dilation
    attr_op_ceil_mode = ceil_mode

    max_out = grad
    max_indices = argmax

    # NHWC -> NCHW
    if input_x_format == "NHWC":
        input_x = input_x.transpose(0, 3, 1, 2)
        max_out = max_out.transpose(0, 3, 1, 2)
        max_indices = max_indices.transpose(0, 3, 1, 2)

    if attr_op_dtype == 3:
        max_indices = max_indices.astype(np.int64)
    if inpu_x_dtype == "bfloat16" or inpu_x_dtype == "float16":
        input_x = input_x.astype(np.float32)
        max_out = max_out.astype(np.float32)
    input_x = torch.from_numpy(input_x)
    max_out = torch.from_numpy(max_out)
    max_indices = torch.from_numpy(max_indices)

    backward_out = torch.ops.aten.max_pool2d_with_indices_backward(
        max_out,
        input_x,
        kernel_size=attr_re_ksize,
        stride=attr_re_strides,
        padding=attr_re_pads,
        dilation=attr_op_dilations,
        ceil_mode=attr_op_ceil_mode,
        indices=max_indices,
    )
    if inpu_x_dtype == "bfloat16" or inpu_x_dtype == "float16":
        backward_out = backward_out.numpy().astype(inpu_x_dtype)
    else:
        backward_out = backward_out.numpy()

    # NCHW -> NHWC
    if input_x_format == "NHWC":
        backward_out = backward_out.transpose(0, 2, 3, 1)

    return backward_out
