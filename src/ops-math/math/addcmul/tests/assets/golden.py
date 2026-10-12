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
import numpy

__golden__ = {
    "kernel": {"addcmul": "addcmul_golden"},
    "aclnn": {"aclnnAddcmul": "aclnn_addcmul_golden"},
}


def addcmul_golden(input_data, x1, x2, value, **kwargs):
    """
    Kernel golden for addcmul.
    All the parameters follow @addcmul_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
              input_formats, output_formats, input_ori_formats, output_ori_formats,
              input_dtypes, output_dtypes.
    """
    import torch
    from ml_dtypes import bfloat16

    data_type = input_data.dtype
    if data_type == bfloat16 or data_type == numpy.float16:
        input_data = torch.from_numpy(input_data.astype(numpy.float32))
        x1 = torch.from_numpy(x1.astype(numpy.float32))
        x2 = torch.from_numpy(x2.astype(numpy.float32))
    else:
        input_data = torch.from_numpy(input_data)
        x1 = torch.from_numpy(x1)
        x2 = torch.from_numpy(x2)
    value = value.item()
    res = torch.addcmul(input_data, x1, x2, value=value)
    res_np = res.numpy()
    res_np = res_np.astype(data_type, copy=False)

    return res_np


def aclnn_addcmul_golden(self, tensor1, tensor2, value, out, **kwargs):
    """
    Aclnn golden for aclnnAddcmul.
    All the parameters (name & order) follow \
        function `aclnnAddcmulGetWorkspaceSize` in @aclnn_addcmul.h \
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

    return torch.addcmul(self, tensor1, tensor2, value=value)
