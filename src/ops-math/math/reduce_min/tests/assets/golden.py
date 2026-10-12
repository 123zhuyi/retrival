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


import torch


__golden__ = {
    "aclnn": {
        "aclnnAminmax": "aclnn_aminmax_golden",
        "aclnnAminmaxDim": "aclnn_aminmax_dim_golden",
        "aclnnAminmaxAll": "aclnn_aminmax_all_golden",
    },
    "kernel": {"reduce_min": "reduce_min_golden"},
}


# def reduce_min_golden(x, axes=None, keep_dims: bool = False, **kwargs):
#     """
#     Kernel golden for reduce_min.
#     """
#     import numpy as np

#     input_dtype = x.dtype
#     if str(input_dtype) == "bfloat16":
#         x = x.astype(np.float32)

#     if axes is not None:
#         axis = tuple(int(a) for a in np.asarray(axes).flatten())
#     else:
#         axis = None

#     try:
#         import tensorflow as tf
#         x_tensor = tf.constant(x)
#         res_tensor = tf.reduce_min(x_tensor, axis=axis, keepdims=keep_dims)
#         res = res_tensor.numpy()
#     except ImportError:
#         try:
#             import torch
#             x_torch = torch.from_numpy(x)
#             if axis is not None:
#                 res_torch = torch.amin(x_torch, dim=axis, keepdim=keep_dims)
#             else:
#                 res_torch = torch.amin(x_torch, keepdim=keep_dims)
#             res = res_torch.numpy()
#         except ImportError:
#             res = np.min(x, axis=axis, keepdims=keep_dims)
#     return res.astype(input_dtype, copy=False)


def aclnn_aminmax_golden(self, dim=0, keepDim=0, minOut=None, maxOut=None, **kwargs):
    """
    Aclnn golden for aclnnAminmax.
    """
    if isinstance(dim, (tuple, list)):
        min_val = torch.amin(self, dim=dim, keepdim=bool(keepDim))
        max_val = torch.amax(self, dim=dim, keepdim=bool(keepDim))
    else:
        result = torch.aminmax(self, dim=dim[0], keepdim=bool(keepDim))
        min_val = result.min
        max_val = result.max
    return [min_val, max_val]


def aclnn_aminmax_all_golden(self, minOut=None, maxOut=None, **kwargs):
    """
    Aclnn golden for aclnnAminmaxAll.
    """
    result = torch.aminmax(self)
    return [result.min, result.max]


def aclnn_aminmax_dim_golden(
    self, dim=0, keepDim=0, minOut=None, maxOut=None, **kwargs
):
    """
    Aclnn golden for aclnnAminmaxDim.
    """
    result = torch.aminmax(self, dim=dim, keepdim=bool(keepDim))
    return [result.min, result.max]
