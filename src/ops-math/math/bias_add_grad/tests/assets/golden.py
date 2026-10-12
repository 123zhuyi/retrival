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

__golden__ = {
    "kernel": {
        "bias_add_grad": "bias_add_grad_golden"
    }
}


def _infer_axes(input_data_format, data_format, shape):
    g_shape_list = []
    if input_data_format == 'FRACTAL_NZ':
        if data_format == "NCHW":
            if len(shape) == 4:
                for i in range(-1 * len(shape), 0):
                    if i not in (-1, -4):
                        g_shape_list += [i + len(shape)]
            elif len(shape) == 5:
                for i in range(-1 * len(shape), 0):
                    if i not in (-2, -3):
                        g_shape_list += [i + len(shape)]
            else:
                g_shape_list.append(0)
                for i in range(2, len(shape)):
                    g_shape_list = g_shape_list + [i]
        else:
            if len(shape) < 4:
                raise RuntimeError("cce_bias_add_grad_nz_2_nhwc only support shape larger than 4D")
            for i in range(-1 * len(shape), 0):
                if i not in (-1, -4):
                    g_shape_list += [i + len(shape)]
    elif input_data_format in ("FRACTAL_Z", "FRACTAL_Z_3D", "NC1HWC0", "NDC1HWC0"):
        if input_data_format == "FRACTAL_Z":
            g_shape_list = [1, 2, 3, 4]
        elif input_data_format == "FRACTAL_Z_3D":
            g_shape_list = [0, 2, 3, 4, 5]
        elif input_data_format == "NC1HWC0":
            g_shape_list = [0, 2, 3]
        elif input_data_format == "NDC1HWC0":
            g_shape_list = [0, 1, 3, 4]
    else:
        if data_format == "NCHW":
            g_shape_list = [0]
            for i in range(2, len(shape)):
                g_shape_list += [i]
        else:
            if len(shape) < 2:
                raise RuntimeError("cce_bias_add_grad only support shape larger than 2D")
            g_shape_list = [x for x in range(len(shape) - 1)]
    return g_shape_list


def __eliminate_duplicate_axes(axis, x):
    axis = tuple(set([_ax if _ax >= 0 else len(x.shape) + _ax for _ax in axis]))
    return axis


def bias_add_grad_golden(x, *, data_format, **kwargs):
    '''
    Golden function for bias_add_grad.
    All the parameters (names and order) follow @bias_add_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    actual_formats = kwargs.get('input_formats', ['ND'])
    axis = __eliminate_duplicate_axes(_infer_axes(actual_formats[0], data_format, x.shape), x)
    return np.sum(x, axis=axis, dtype="float64")
