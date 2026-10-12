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
        "batch_to_space": "batch_to_space_golden"
    }
}


def batch_to_space_golden(x, crops, *, block_size, **kwargs):
    """
    Golden function for BatchToSpace (inverse of SpaceToBatch).

    x:         4D NHWC [N * bs * bs, H_in, W_in, C]
    crops:     1D [crop_top, crop_bottom, crop_left, crop_right]
    block_size: int

    Returns output [N, H_out, W_out, C]
    """
    N_in, H_in, W_in, C = x.shape
    bs = block_size
    crop_top = int(crops[0])
    crop_bottom = int(crops[1])
    crop_left = int(crops[2])
    crop_right = int(crops[3])

    N = N_in // (bs * bs)

    # Reshape batch into spatial blocks
    # [N_in, H_in, W_in, C] -> [bs, bs, N, H_in, W_in, C]
    out = x.reshape(bs, bs, N, H_in, W_in, C)
    # -> [N, H_in, bs, W_in, bs, C]
    out = out.transpose(2, 3, 0, 4, 1, 5)
    # -> [N, H_in * bs, W_in * bs, C]
    out = out.reshape(N, H_in * bs, W_in * bs, C)

    # Crop spatial boundaries
    H_end = out.shape[1] - crop_bottom
    W_end = out.shape[2] - crop_right
    out = out[:, crop_top:H_end, crop_left:W_end, :]

    return out
