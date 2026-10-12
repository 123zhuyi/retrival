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
from functools import reduce
import operator

__input__ = {
    "kernel": {"max_pool_grad_with_argmax_v3": "max_pool_grad_with_argmax_v3_input"}
}


def max_pool_grad_with_argmax_v3_input(
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
    Input function for max_pool_grad_with_argmax_v3.
    All the parameters (names and order) follow @max_pool_grad_with_argmax_v3_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  input_ranges, full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors (length must match Input count in _def.cpp)
    """
    import torch
    import torch.nn.functional as F

    input_dytpe_source = kwargs["input_dtypes"][0]
    if kwargs["input_dtypes"][0] == "bool":
        input_type = np.bool_
    elif kwargs["input_dtypes"][0] == "bfloat16" or input_dytpe_source == "float16":
        input_type = np.float32
    else:
        input_type = getattr(np, kwargs["input_dtypes"][0])

    input_x_format = kwargs["input_formats"][0]
    input_shape = np.s_[x.shape]
    ele_num = reduce(operator.mul, input_shape)
    random_array = np.random.randint(
        low=np.iinfo(np.int8).min,
        high=np.iinfo(np.int8).max + 1,
        size=(ele_num,),
        dtype=np.int8,
    )
    input_x = random_array.reshape(input_shape).astype(input_type)
    input_x = torch.from_numpy(input_x)

    attr_re_ksize = ksize
    attr_re_strides = strides
    attr_re_pads = pads
    attr_op_dtype = dtype
    attr_op_dilations = dilation
    attr_op_ceil_mode = ceil_mode

    # NHWC -> NCHW
    if input_x_format == "NHWC":
        input_x = input_x.permute(0, 3, 1, 2)

    max_out, max_indices = F.max_pool2d_with_indices(
        input_x,
        kernel_size=attr_re_ksize,
        stride=attr_re_strides,
        padding=attr_re_pads,
        dilation=attr_op_dilations,
        ceil_mode=attr_op_ceil_mode,
    )

    if input_dytpe_source == "bfloat16" or input_dytpe_source == "float16":
        input_x = input_x.numpy().astype(input_dytpe_source)
        max_out = max_out.numpy().astype(input_dytpe_source)
    else:
        input_x = input_x.numpy()
        max_out = max_out.numpy()
    max_indices = max_indices.numpy().astype(kwargs["input_dtypes"][2])

    if attr_op_dtype == 3:
        max_indices = max_indices.astype(np.int32)
    elif attr_op_dtype == 9:
        max_indices = max_indices.astype(np.int64)

    # NCHW -> NHWC
    if input_x_format == "NHWC":
        input_x = input_x.transpose(0, 2, 3, 1)
        max_out = max_out.transpose(0, 2, 3, 1)
        max_indices = max_indices.transpose(0, 2, 3, 1)

    return [input_x, max_out, max_indices]
