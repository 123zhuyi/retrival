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


__golden__ = {"kernel": {"reduce_log_sum_exp": "reduce_log_sum_exp_golden"}}


def reduce_log_sum_exp_golden(x, axes=None, keep_dims: bool = False, **kwargs):
    """
    Kernel golden for reduce_log_sum_exp.
    All the parameters follow @reduce_log_sum_exp_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import numpy as np
    import torch

    input_dtype = x.dtype
    if str(input_dtype) == "bfloat16":
        x = x.astype(np.float32)

    # Convert numpy array to torch tensor
    x_torch = torch.from_numpy(x)

    if axes is not None:
        axis_tuple = tuple(int(a) for a in np.asarray(axes).flatten())
        res_torch = torch.logsumexp(x_torch, dim=axis_tuple, keepdim=keep_dims)
    else:
        # Reduce over all dimensions - flatten first
        res_torch = torch.logsumexp(x_torch.flatten(), dim=0, keepdim=keep_dims)

    # Convert back to numpy
    res = res_torch.numpy()

    return res.astype(input_dtype, copy=False)
