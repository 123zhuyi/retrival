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
"""SpaceToBatchND golden reference implementation (vectorized)."""

__golden__ = {
    "kernel": {
        "space_to_batch_nd": "space_to_batch_nd_golden"
    }
}

__input__ = {
    "kernel": {
        "space_to_batch_nd": "space_to_batch_nd_input"
    }
}

import numpy as np
import itertools


def space_to_batch_nd_input(x, block_shape, paddings, **kwargs):
    x_flat = np.arange(x.size, dtype=np.float64)
    x[:] = x_flat.reshape(x.shape).astype(x.dtype)
    return x, block_shape, paddings


def space_to_batch_nd_golden(x, block_shape, paddings, **extra):
    """
    Golden reference for SpaceToBatchND operator.

    Args:
        x: Input ND tensor, shape [B, S1, ..., SN, T1, ..., Tk]
        block_shape: 1D int tensor, shape [N]
        paddings: 2D int tensor, shape [N, 2]
        **extra: extended parameters (unused)

    Returns:
        (y,) where y is the output ND tensor
    """
    rank = len(x.shape)
    N = int(np.prod(block_shape.shape))
    dtype = x.dtype

    bs = block_shape.flatten().astype(np.int64)
    pd = paddings.reshape(N, 2).astype(np.int64)

    batch_mul = int(np.prod(bs))

    # Output shape
    out_shape = list(x.shape)
    out_shape[0] = int(x.shape[0]) * batch_mul
    for i in range(N):
        si = int(x.shape[1 + i])
        out_shape[1 + i] = (si + int(pd[i, 0]) + int(pd[i, 1])) // int(bs[i])
    out_shape = tuple(out_shape)

    y = np.zeros(out_shape, dtype=dtype)

    # Build block index grid: iterate over all (b0, ..., bN-1) combinations
    block_ranges = [range(int(bs[i])) for i in range(N)]
    batch_stride = batch_mul

    for block_indices in itertools.product(*block_ranges):
        # Compute block offset in output batch dimension
        block_off = 0
        for i in range(N):
            if i < N - 1:
                stride_i = int(np.prod(bs[i + 1:]))
            else:
                stride_i = 1
            block_off += block_indices[i] * stride_i

        # Build input slice and output slice
        x_slices = [slice(None)]  # batch dim: all
        batch = int(x.shape[0])
        y_slices = [slice(block_off * batch, block_off * batch + batch)]  # output batch: consecutive

        for i in range(N):
            bi = block_indices[i]
            bsi = int(bs[i])
            pt = int(pd[i, 0])
            pb = int(pd[i, 1])
            si = int(x.shape[1 + i])
            soi = out_shape[1 + i]

            # First valid output spatial index: o * bsi + bi >= pt
            out_first = max(0, (pt - bi + bsi - 1) // bsi)
            # Last valid output spatial index (exclusive): o * bsi + bi < pt + si
            out_last = min(soi, (si + pt - bi + bsi - 1) // bsi)

            if out_first >= out_last:
                x_slices.append(slice(0, 0))
                y_slices.append(slice(0, 0))
            else:
                start_in = out_first * bsi + bi - pt
                in_end = out_last * bsi + bi - pt
                if in_end > si:
                    in_end = si
                x_slices.append(slice(start_in, in_end, bsi))
                y_slices.append(slice(out_first, out_last))

        # Remaining dims unchanged
        for i in range(N + 1, rank):
            x_slices.append(slice(None))
            y_slices.append(slice(None))

        y[tuple(y_slices)] = x[tuple(x_slices)]

    return y