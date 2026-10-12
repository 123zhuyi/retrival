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
        "clip_by_value_v2": "clip_by_value_v2_golden"
    }
}


def clip_by_value_v2_golden(x, clip_value_min, clip_value_max, **kwargs):
    '''
    Golden function for clip_by_value_v2.
    All the parameters (names and order) follow @clip_by_value_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    if "bfloat16" in str(x.dtype):
        x = x.astype("float32")
        clip_value_min = clip_value_min.astype("float32")
        clip_value_max = clip_value_max.astype("float32")
    max_ = np.maximum(x, clip_value_min)
    res = np.minimum(max_, clip_value_max)
    if "bfloat16" in str(x.dtype):
        return res.astype(x.dtype, copy=False)
    return res
