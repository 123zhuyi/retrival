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
    "kernel": {
        "acts_ulq_input_grad": "acts_ulq_input_grad_golden",
    }
}


def acts_ulq_input_grad_golden(
    y_grad,
    clamp_min_mask,
    clamp_max_mask,
    **kwargs,
):
    """
    Kernel golden for acts_ulq_input_grad.
    All the parameters follow @acts_ulq_input_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
                        input_formats, output_formats, input_ori_formats,
                        output_ori_formats, input_dtypes, output_dtypes.
    """
    del kwargs

    y_grad_tensor = torch.from_numpy(y_grad)
    clamp_min_mask_tensor = torch.from_numpy(clamp_min_mask)
    clamp_max_mask_tensor = torch.from_numpy(clamp_max_mask)
    signal = torch.mul(clamp_min_mask_tensor, clamp_max_mask_tensor)
    return torch.mul(y_grad_tensor, signal).numpy()
