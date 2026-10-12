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

__input__ = {"kernel": {"segment_sum": "segment_sum_input"}}


def segment_sum_input(x, segment_ids, **kwargs):
    """
    Input function for segment_sum.
    All the parameters (names and order) follow @segment_sum_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  input_ranges, full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors
    """
    ids_dtype = kwargs["input_dtypes"][1]

    lower_limit = 0
    upper_limit = 1
    output_ori_shapes = kwargs.get("output_ori_shapes")
    if output_ori_shapes and len(output_ori_shapes) > 0:
        out_shape = output_ori_shapes[0]
        if out_shape and len(out_shape) > 0:
            upper_limit = out_shape[0]

    if upper_limit < 1:
        upper_limit = 1

    new_ids = np.random.uniform(lower_limit, upper_limit, segment_ids.shape).astype(
        ids_dtype
    )
    new_ids = np.sort(new_ids)
    if new_ids.size > 0:
        new_ids[-1] = upper_limit - 1

    return [x, new_ids]
