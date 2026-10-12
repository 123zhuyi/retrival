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
    "kernel": {"max_pool3d_with_argmax_v2": "max_pool3d_with_argmax_v2_golden"}
}


def max_pool3d_with_argmax_v2_golden(
    x,
    *,
    ksize,
    strides,
    pads,
    dilation=[1, 1, 1],
    ceil_mode=False,
    data_format="NCDHW",
    dtype=3,
    **kwargs,
):
    """
    Golden function for max_pool3d_with_argmax_v2.
    All the parameters (names and order) follow @max_pool3d_with_argmax_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor (y, argmax)
    """
    import torch
    import torch.nn as nn

    input_x = deepcopy(x)
    input_x_format = kwargs["input_formats"][0]
    input_x_dtype = input_x.dtype

    out_y_format = kwargs["output_formats"][0]
    out_argmax_format = kwargs["output_formats"][1]

    attr_re_ksize = ksize
    if len(attr_re_ksize) == 3:
        attr_re_ksize = [attr_re_ksize[0], attr_re_ksize[1], attr_re_ksize[2]]

    attr_re_strides = strides
    if len(attr_re_strides) == 3:
        attr_re_strides = [attr_re_strides[0], attr_re_strides[1], attr_re_strides[2]]

    attr_re_pads = pads
    if len(attr_re_pads) == 3:
        attr_re_pads = [attr_re_pads[0], attr_re_pads[1], attr_re_pads[2]]

    attr_op_dtype = dtype
    attr_op_dilations = dilation
    if attr_op_dilations is not None and len(attr_op_dilations) == 3:
        attr_op_dilations = [
            attr_op_dilations[0],
            attr_op_dilations[1],
            attr_op_dilations[2],
        ]

    attr_op_ceil_mode = ceil_mode
    attr_op_format = data_format

    input_x_format = attr_op_format

    # NDHWC -> NCDHW (if needed)
    if input_x_format == "NDHWC":
        input_x = input_x.transpose(0, 4, 1, 2, 3)

    # Convert to float32 if input is float16/bfloat16
    if str(input_x_dtype) in ("float16", "bfloat16"):
        input_x = input_x.astype(np.float32)

    input_x = torch.from_numpy(input_x)

    # Torch MaxPool3d setup
    attr = {
        "kernel_size": attr_re_ksize,
        "stride": attr_re_strides,
        "padding": attr_re_pads,
        "return_indices": True,
    }
    if attr_op_dilations is not None:
        attr["dilation"] = attr_op_dilations
    if attr_op_ceil_mode is not None:
        attr["ceil_mode"] = attr_op_ceil_mode

    cpuMaxPool3d = nn.MaxPool3d(**attr)
    max_out, max_indices = cpuMaxPool3d(input_x)

    if str(input_x_dtype) == "bfloat16":
        out_y = max_out.numpy().astype(input_x_dtype, copy=False)
    else:
        out_y = max_out.numpy().astype(input_x_dtype)
    if out_y_format == "NDHWC":
        out_y = out_y.transpose(0, 2, 3, 4, 1)

    if attr_op_dtype == 3 or attr_op_dtype is None:
        out_argmax = max_indices.numpy().astype(np.int32)
    elif attr_op_dtype == 9:
        out_argmax = max_indices.numpy().astype(np.int64)
    else:
        out_argmax = max_indices.numpy()

    if out_argmax_format == "NDHWC":
        out_argmax = out_argmax.transpose(0, 2, 3, 4, 1)
    return out_y, out_argmax
