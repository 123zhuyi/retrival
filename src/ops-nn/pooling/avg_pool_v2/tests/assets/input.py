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


__input__ = {"kernel": {"avg_pool_v2": "avg_pool_v2_input"}}


def avg_pool_v2_input(
    x,
    ksize,
    strides,
    padding_mode="CALCULATED",
    pads=(0, 0, 0, 0),
    data_format="NCHW",
    global_pooling=False,
    ceil_mode=False,
    exclusive=True,
    divisor_override=0,
    **kwargs,
):
    """
    Input function for avg_pool_v2.
    All the parameters follow @avg_pool_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    """
    return (x,)
