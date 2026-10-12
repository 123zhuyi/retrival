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

import numpy

__golden__ = {
    "kernel": {
        "depth_to_space": "depth_to_space_golden"
    }
}


def depth_to_space_golden(x, *, block_size, mode, data_format, **kwargs):
    '''
    Golden function for depth_to_space.
    All the parameters (names and order) follow @depth_to_space_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    shapes = x.shape

    if data_format == "NCHW":
        n, c, h, w = shapes
        output_shapes = [n, c // (block_size ** 2), h * block_size, w * block_size]
        if mode == "CRD":
            input_shapes = [n, c // (block_size ** 2), block_size, block_size, h, w]
            perm = [0, 1, 4, 2, 5, 3]
        else:  # mode == "DCR"
            input_shapes = [n, block_size, block_size, c // (block_size ** 2), h, w]
            perm = [0, 3, 4, 1, 5, 2]
    else:  # data_format == "NHWC":
        n, h, w, c = shapes
        output_shapes = [n, h * block_size, w * block_size, c // (block_size ** 2)]
        if mode == "CRD":
            input_shapes = [n, h, w, c // (block_size ** 2), block_size, block_size]
            perm = [0, 1, 4, 2, 5, 3]
        else:  # mode == "DCR"
            input_shapes = [n, h, w, block_size, block_size, c // (block_size ** 2)]
            perm = [0, 1, 3, 2, 4, 5]
    tmp = x.reshape(input_shapes)
    tmp = numpy.transpose(tmp, perm)
    return tmp.reshape(output_shapes)
