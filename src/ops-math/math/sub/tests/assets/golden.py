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

__golden__ = {
    "kernel": {"sub": "sub_golden"},
    "aclnn": {"aclnnSub": "aclnn_sub_golden"},
}


def sub_golden(x1, x2, **kwargs):
    """
    Kernel golden for sub.
    All the parameters follow @sub_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
            input_formats, output_formats, input_ori_formats, output_ori_formats,
            input_dtypes, output_dtypes.
    """
    import torch

    dtype = x1.dtype
    if "bfloat16" in str(dtype):
        x1 = x1.astype("float32")
        x2 = x2.astype("float32")
    x = torch.from_numpy(x1)
    y = torch.from_numpy(x2)
    if "bool" in str(dtype):
        res = torch.logical_xor(x, y).numpy()
    else:
        res = torch.sub(x, y).numpy()
    if "bfloat16" in str(dtype):
        res = res.astype(dtype)

    return res


def aclnn_sub_golden(self, other, alpha, out, **kwargs):
    """
    Aclnn golden for aclnnSub.
    All the parameters (name & order) follow \
        function `aclnnSubGetWorkspaceSize` in @aclnn_sub.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    """
    import torch

    return torch.sub(self, other, alpha=alpha)
