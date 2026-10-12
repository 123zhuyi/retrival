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

__golden__ = {"kernel": {"prelu": "prelu_golden"}}


def _broadcast_to_maxshape(shapes):
    """
    produce broadcast shape
    for example:
        input: shape is [[2, 3], [3, 2, 1], [3, 1, 3]]
        output: [1, 2, 3], [3, 2, 1], [3, 1, 3], [3, 2, 3]
    """

    def _max(_shape):
        no_one_shape = [s for s in _shape if s != 1]
        if len(no_one_shape) == 0:
            max_value = 1
        else:
            max_value = no_one_shape[0]
        return max_value

    max_dim_length = max(len(list(shape)) for shape in shapes)
    input_shapes = []
    for shape in shapes:
        input_shapes.append(
            [1 for _ in range(max_dim_length - len(shape))] + list(shape)
        )
    input_shapes = list(map(list, zip(*input_shapes)))
    max_shape = [_max(shape) for shape in input_shapes]
    input_shapes = list(map(list, zip(*input_shapes)))
    return (*input_shapes, max_shape)


def _broadcast_inputs_shape(x, weight, format_):
    shape_x = x.shape
    shape_w = weight.shape
    x_dim = len(shape_x)
    w_dim = len(shape_w)
    if format_ == "NC1HWC0":
        if w_dim != 5:
            if w_dim == 1:
                weight_shape_new = [1] * 5
            else:
                shape_list = _broadcast_to_maxshape([shape_x, shape_w])
                shape_x, weight_shape_new = shape_list[0], shape_list[1]
        else:
            c1 = shape_x[1]
            c0 = shape_x[4]
            weight_shape_new = [1, c1, 1, 1, c0]
    elif format_ == "NDC1HWC0":
        if w_dim != 6:
            if w_dim == 1:
                weight_shape_new = [1] * 6
            else:
                shape_list = _broadcast_to_maxshape([shape_x, shape_w])
                shape_x, weight_shape_new = shape_list[0], shape_list[1]
        else:
            c1 = shape_x[2]
            c0 = shape_x[5]
            weight_shape_new = [1, 1, c1, 1, 1, c0]
    elif format_ == "FRACTAL_NZ":
        if w_dim == 1 and shape_w[0] == 1:
            weight_shape_new = [1] * x_dim
        elif w_dim == 1:
            weight_shape_new = [1] * x_dim
            weight_shape_new[0] = shape_x[0]
            weight_shape_new[-1] = shape_x[-1]
        else:
            shape_list = _broadcast_to_maxshape([shape_x, shape_w])
            shape_x, weight_shape_new = shape_list[0], shape_list[1]
    elif format_ == "NHWC" and x_dim == 4:
        if (w_dim == 1 and shape_w[0] != shape_x[-1] and shape_w[0] != 1) or (
            w_dim not in (1, 3)
        ):
            shape_list = _broadcast_to_maxshape([shape_x, shape_w])
            shape_x, weight_shape_new = shape_list[0], shape_list[1]
        elif w_dim == 1:
            weight_shape_new = [1] * x_dim
            weight_shape_new[3] = shape_x[-1]
        else:
            weight_shape_new = list(shape_w)
            weight_shape_new.insert(0, 1)
    elif x_dim == 1:
        if shape_w[0] != 1 or w_dim != 1:
            shape_list = _broadcast_to_maxshape([shape_x, shape_w])
            shape_x, weight_shape_new = shape_list[0], shape_list[1]
        else:
            weight_shape_new = [1]
    else:
        if (shape_w[0] != shape_x[1] and shape_w[0] != 1) or (
            w_dim not in (1, x_dim - 1)
        ):
            shape_list = _broadcast_to_maxshape([shape_x, shape_w])
            shape_x, weight_shape_new = shape_list[0], shape_list[1]
        elif w_dim == 1:
            weight_shape_new = [1] * x_dim
            if shape_w[0] != 1:
                weight_shape_new[1] = shape_w[0]
        elif w_dim == x_dim - 1:
            weight_shape_new = list(shape_w)
            weight_shape_new.insert(0, 1)

    return shape_x, weight_shape_new


def prelu_golden(x, weight, **kwargs):
    """
    Golden function for prelu.
    All the parameters (names and order) follow @prelu_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    x_format = kwargs["input_formats"][0]
    weight_format = kwargs["input_formats"][1]
    shape_x, shape_weight = _broadcast_inputs_shape(x, weight, x_format)
    input_x = x.reshape(shape_x)
    weight = weight.reshape(shape_weight)

    if x_format == "FRACTAL_NZ" and weight_format != "FRACTAL_NZ":
        target_shape = [1] * len(shape_x)
        if sum(shape_weight) != 1:
            target_shape[0] = shape_x[0]
            target_shape[-1] = shape_x[-1]
        weight = np.reshape(weight, target_shape)

    shape_list = _broadcast_to_maxshape([shape_x, shape_weight])
    input_x = np.broadcast_to(input_x, shape_list[2])
    weight_input = np.broadcast_to(weight, shape_list[2])

    return np.where(input_x > 0, input_x, weight_input * input_x)
