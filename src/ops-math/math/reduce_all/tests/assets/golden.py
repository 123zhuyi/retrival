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


__golden__ = {"kernel": {"reduce_all": "reduce_all_golden"}}


def reduce_all_golden(x, axes=None, keep_dims: bool = False, **kwargs):
    """
    Kernel golden for reduce_all.
    All the parameters follow @reduce_all_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import numpy as np

    input_dtype = x.dtype

    if axes is not None:
        axis_tuple = tuple(int(a) for a in np.asarray(axes).flatten())
    else:
        axis_tuple = None

    res = np.all(x, axis=axis_tuple, keepdims=keep_dims)

    return res.astype(input_dtype, copy=False)
