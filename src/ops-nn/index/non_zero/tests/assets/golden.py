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

__golden__ = {"kernel": {"non_zero": "non_zero_golden"}}


def non_zero_golden(x, *, transpose=False, **kwargs):
    """
    Golden function for non_zero.
    All the parameters (names and order) follow @non_zero_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    output_dtype = kwargs["output_dtypes"][0]

    if x.dtype.name == "bfloat16":
        import torch

        np_int16 = x.view(dtype=np.int16)
        x_torch = torch.from_numpy(np_int16).view(torch.bfloat16)
        if transpose:
            ret_torch = torch.stack(torch.nonzero(x_torch, as_tuple=True))
        else:
            ret_torch = torch.nonzero(x_torch, as_tuple=False)
        ret = ret_torch.numpy()
    else:
        if transpose:
            ret = np.asarray(np.nonzero(x))
        else:
            ret = np.transpose(np.nonzero(x))

    return ret.astype(output_dtype, copy=False)
