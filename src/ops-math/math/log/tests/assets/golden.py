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
import torch

__golden__ = {
    "aclnn": {
        "aclnnLog": "aclnn_log_golden",
    },
    "kernel": {"log": "log_golden"},
}


def log_golden(x, base: float = -1.0, scale: float = 1.0, shift: float = 0.0, **kwargs):
    """
    Kernel golden for log.
    All the parameters follow @log_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
              input_formats, output_formats, input_ori_formats, output_ori_formats,
              input_dtypes, output_dtypes.
    """
    x_dtype = x.dtype
    if x_dtype.name == "bfloat16" or x_dtype.name == "float16":
        x = torch.from_numpy(x.astype(np.float32))
    else:
        x = torch.from_numpy(x)

    if scale == 1 and shift == 0:
        if base == -1:
            output = torch.log(x)
        elif base == 2:
            output = torch.log2(x)
        elif base == 10:
            output = torch.log10(x)
        else:
            output = torch.log((scale * x + shift)) / np.log(base)
    elif base == -1:
        output = torch.log((scale * x + shift))
    else:
        output = torch.log((scale * x + shift)) / np.log(base)

    return output.numpy().astype(x_dtype, copy=False)


def aclnn_log_golden(self, out=None, **kwargs):
    """
    Aclnn golden for aclnnLog.
    Parameters follow @aclnnLogGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch

    return [torch.log(self)]
