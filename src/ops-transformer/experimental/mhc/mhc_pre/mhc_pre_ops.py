#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the
# License.
"""mhc_pre operator: Reduce N streams to 1 via weighted sum.

This module provides PyTorch interface to the AscendC mhc_pre kernel.

Usage:
    import mhc_pre_ops
    output = mhc_pre_ops.mhc_pre(input_tensor, h_pre_weights)
"""

import logging

import torch
import torch_npu

logger = logging.getLogger(__name__)

try:
    import mhc_pre_ext
    _USE_CPP_EXT = True
except ImportError:
    _USE_CPP_EXT = False
    logger.warning("mhc_pre_ext not found. Run 'python setup.py build_ext --inplace' to build.")


def mhc_pre(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    """Reduce N streams to 1 via weighted sum.

    Mathematical operation:
        out[b, s, d] = sum_n(x[b*N + n, s, d] * h_pre[n])

    Equivalent einsum:
        x_4d = x.view(batch, num_streams, seq_len, dim)
        out = torch.einsum('bnsd,n->bsd', x_4d, h_pre)

    Args:
        x: Input tensor [batch * num_streams, seq_len, dim]
        h_pre: Weight tensor [num_streams]

    Returns:
        Output tensor [batch, seq_len, dim]
    """
    if not _USE_CPP_EXT:
        raise RuntimeError("mhc_pre_ext not available. Build with setup.py first.")

    return mhc_pre_ext.forward(x.contiguous(), h_pre.contiguous())


def mhc_pre_einsum(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    """Reference implementation using torch.einsum."""
    num_streams = h_pre.size(0)
    if num_streams == 0:
        raise ValueError("num_streams must be > 0, got 0")
    batch = x.size(0) // num_streams
    seq_len = x.size(1)
    dim = x.size(2)
    x_4d = x.view(batch, num_streams, seq_len, dim)
    return torch.einsum('bnsd,n->bsd', x_4d, h_pre)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    torch.npu.set_device(0)

    batch, seq_len, dim, num_streams = 2, 16, 32, 4
    x = torch.randn(batch * num_streams, seq_len, dim, dtype=torch.float32).npu()
    h = torch.randn(num_streams, dtype=torch.float32).npu()

    out_npu = mhc_pre(x, h)
    out_ref = mhc_pre_einsum(x, h)

    logger.info("Input: %s, h_pre: %s", x.shape, h.shape)
    logger.info("Output: %s", out_npu.shape)
    logger.info("Match: %s", torch.allclose(out_npu, out_ref, atol=1e-5))
    logger.info("Max diff: %.2e", (out_npu - out_ref).abs().max().item())
