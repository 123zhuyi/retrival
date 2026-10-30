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
        "concat_d": "concat_d_golden"
    }
}


def update_axis_for_hw_inner_format(ori_shape, axis, input_format, ori_format, reduce_mode=False):
    if input_format in ("NDC1HWC0", "NC1HWC0"):
        ori_shape_len = len(ori_shape) if -2 not in ori_shape else len(ori_format)
        axis = axis % ori_shape_len
        offset_6hd = 1 if input_format == "NDC1HWC0" else 0
        format_c_axis = 1 + offset_6hd if not reduce_mode else [1 + offset_6hd, 4 + offset_6hd]
        format_axis_map = {
            "N": 0,
            "C": format_c_axis,
            "H": 2 + offset_6hd,
            "W": 3 + offset_6hd,
            "D": 1
        }
        concat_dim_name = ori_format[axis]
        axis = format_axis_map[concat_dim_name]

    if input_format in ("FRACTAL_NZ",):
        axis = axis % len(ori_shape)
        if axis == len(ori_shape) - 1:
            axis = len(ori_shape) - 2 if not reduce_mode else [len(ori_shape) - 2, len(ori_shape) + 1]
        elif axis == len(ori_shape) - 2:
            axis = len(ori_shape) - 1 if not reduce_mode else [len(ori_shape) - 1, len(ori_shape) + 0]

    if input_format in ("FRACTAL_Z", "FRACTAL_Z_3D"):
        axis = axis % len(ori_shape)
        offset_3d = 1 if input_format == "FRACTAL_Z_3D" else 0
        format_c_axis = 0 + offset_3d if not reduce_mode else [0 + offset_3d, 5 + offset_3d]
        format_n_axis = 3 + offset_3d if not reduce_mode else [3 + offset_3d, 4 + offset_3d]
        format_axis_map = {
            "N": format_n_axis,
            "C": format_c_axis,
            "H": 1 + offset_3d,
            "W": 2 + offset_3d,
            "D": 0
        }
        concat_dim_name = ori_format[axis]
        axis = format_axis_map[concat_dim_name]

    return axis


def concat_d_golden(x, *, concat_dim, N=1, **kwargs):
    '''
    Golden function for concat.
    All the parameters (names and order) follow @concat_d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    x_arrays = list(x)

    ori_shape = kwargs.get('input_ori_shapes', [x[0].shape])[0]
    input_formats = kwargs.get('input_formats', ['ND'])
    input_ori_formats = kwargs.get('input_ori_formats', ['ND'])
    
    concat_dim = update_axis_for_hw_inner_format(ori_shape, concat_dim, input_formats[0], input_ori_formats[0])
    return numpy.concatenate(x_arrays, axis=concat_dim)
