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

__golden__ = {
    "kernel": {
        "space_to_batch": "space_to_batch_golden"
    }
}


def space_to_batch_golden(x, paddings, *, block_size, **kwargs):
    """
    Golden function for SpaceToBatch.

    x:         4D NHWC [N, H_in, W_in, C]
    paddings:  1D [pad_top, pad_bottom, pad_left, pad_right]
    block_size: int

    Returns output [N * bs * bs, H_out, W_out, C]
    """
    N, H_in, W_in, C = x.shape
    bs = block_size
    pad_top = int(paddings[0])
    pad_bottom = int(paddings[1])
    pad_left = int(paddings[2])
    pad_right = int(paddings[3])

    H_padded = H_in + pad_top + pad_bottom
    W_padded = W_in + pad_left + pad_right
    H_out = H_padded // bs
    W_out = W_padded // bs

    # Spatial padding: pad H and W dimensions with zeros
    out = np.pad(
        x,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )
    # [N, H_padded, W_padded, C]

    # Reshape spatial blocks into batch dimension
    # [N, H_out, bs, W_out, bs, C]
    out = out.reshape(N, H_out, bs, W_out, bs, C)
    # -> [bs, bs, N, H_out, W_out, C]
    out = out.transpose(2, 4, 0, 1, 3, 5)
    # -> [N * bs * bs, H_out, W_out, C]
    out = out.reshape(N * bs * bs, H_out, W_out, C)

    return out
