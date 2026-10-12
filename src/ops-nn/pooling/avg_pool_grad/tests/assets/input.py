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


__input__ = {"kernel": {"avg_pool_grad": "avg_pool_grad_input"}}


def avg_pool_grad_input(
    orig_input_shape, input_grad, ksize, strides, padding, data_format="NHWC", **kwargs
):
    """
    Input function for avg_pool_grad.
    All the parameters follow @avg_pool_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    """
    input_dtypes = kwargs.get("input_dtypes")
    orig_input = numpy.array(orig_input_shape, input_dtypes[0])
    return (orig_input, input_grad)
