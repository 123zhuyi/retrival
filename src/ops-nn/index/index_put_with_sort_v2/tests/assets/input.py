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

__input__ = {"kernel": {"index_put_with_sort_v2": "index_put_with_sort_v2_input"}}


def prod_func(src_list):
    res = 1
    for i in src_list:
        res *= i
    return res


def product_of_subsequent_elements(x):
    """
    返回一个数组，其中每个元素是原数组中相应位置之后的元素的乘积

    参数:
    x -- 输入数组

    返回:
    一个与x长度相同的新数组
    """
    result = [1] * len(x)  # 初始化结果数组，所有元素为1
    product = 1

    # 从右向左计算累积乘积
    for i in range(len(x) - 1, -1, -1):
        result[i] = product
        product *= x[i]

    return result


def index_put_with_sort_v2_input(
    self, linear_index, pos_idx, values, *, indexed_sizes, accumulate=False, **kwargs
):
    """
    Input function for index_put_with_sort_v2.
    All the parameters (names and order) follow @index_put_with_sort_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  input_ranges, full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors (length must match Input count in _def.cpp)
    """
    x = self
    value_shape = values.shape
    x_shape = x.shape
    x = np.zeros(x.shape).astype(x.dtype)

    index_dtype = linear_index.dtype
    full_stride = product_of_subsequent_elements(x_shape)
    mask = indexed_sizes
    np.random.seed(0)

    origin_indices_list = []
    input_stride = []
    for i in range(len(x_shape)):
        if mask[i] != 0:
            origin_indices_list.append(
                np.random.randint(x_shape[i], size=linear_index.size).astype(
                    index_dtype
                )
            )
            input_stride.append(full_stride[i])
    origin_linear_index = [0] * linear_index.size  # 拉平一维
    for i in range(len(origin_indices_list)):
        origin_linear_index += origin_indices_list[i] * input_stride[i]

    linear_index = np.sort(origin_linear_index).astype(index_dtype)
    pos_idx = np.argsort(origin_linear_index).astype(np.int32)

    indices = np.lexsort((pos_idx, linear_index))
    linear_index = linear_index[indices]
    pos_idx = pos_idx[indices]

    if "float32" in str(x.dtype):
        value = np.random.random(value_shape).astype(x.dtype)
    else:
        value = np.random.randint(10, size=value_shape).astype(x.dtype)

    return [x, linear_index, pos_idx, value]
