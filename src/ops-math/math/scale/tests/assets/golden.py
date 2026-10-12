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
"""
aclnnScale golden: y = x * scale [+ bias]

scale/bias broadcast against x at [axis : axis+span], rest dims size 1.
scaleFromBlob=True : span = numAxes (numAxes==-1 -> to last axis; 0 -> scale shape [1])
scaleFromBlob=False: span = rank(scale) (numAxes ignored).
"""

import torch


__golden__ = {"aclnn": {"aclnnScale": "scale_golden"}}


def _align(x_shape, axis, span, tensor):
    rank = len(x_shape)
    if axis < 0:
        axis += rank
    bshape = [1] * rank
    flat = tensor.reshape(-1)
    bshape[axis : axis + span] = list(x_shape[axis : axis + span])
    if torch.Size(bshape).numel() != flat.numel():
        bshape = [1] * rank
    return flat.reshape(bshape)


def scale_golden(x, scale, bias, axis, numAxes, scaleFromBlob, y=None, **kwargs):
    """
    Aclnn golden for aclnnScale.
    Parameters follow @aclnnScaleGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.

    kwargs may contain: tensor_dtypes, tensor_formats, scalar_dtypes,
                        use_torch, short_soc_version, testcase_name.
    """
    del y, kwargs
    output_dtype = x.dtype
    x = x.to(torch.float32)
    scale = scale.to(torch.float32)
    bias = bias.to(torch.float32) if bias is not None else None
    if bias is not None and bias.numel() == 0:
        bias = None  # 空 bias 张量(shape含0)视为无 bias

    axis = int(axis)
    num_axes = int(numAxes)
    from_blob = bool(scaleFromBlob)

    rank = x.dim()
    a = axis + rank if axis < 0 else axis
    if from_blob:
        span = (rank - a) if num_axes == -1 else num_axes
    else:
        span = scale.dim()
    span = max(0, min(span, rank - a))

    s = _align(x.shape, a, span, scale)
    result = torch.mul(x, s)
    if bias is not None:
        result = torch.add(result, _align(x.shape, a, span, bias))
    return result.to(output_dtype)
