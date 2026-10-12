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
        "cumsum": "cumsum_golden"
    }
}

def cumsum_golden(x, axis, *, exclusive, reverse, **kwargs):
    '''
    Kernel golden for cumsum.
    All the parameters follow @cumsum_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    '''
    import numpy as np

    axis = int(axis.item()) if hasattr(axis, 'item') else int(axis)
    ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
    input_x_dtype = x.dtype
    if ori_dtype and ("bfloat16" in str(ori_dtype).lower() or "float16" in str(ori_dtype).lower()):
        x = x.astype("float32")

    if reverse:
        x = np.flip(x, axis=axis)
    res = np.cumsum(x, axis=axis)
    if exclusive:
        res = np.roll(res, 1, axis=axis)
        res_slice = [slice(None)] * res.ndim
        res_slice[axis] = 0
        res[tuple(res_slice)] = 0
    if reverse:
        res = np.flip(res, axis=axis)

    if ori_dtype and ("bfloat16" in str(ori_dtype).lower() or "float16" in str(ori_dtype).lower()):
        res = res.astype(input_x_dtype, copy=False)
    return res
