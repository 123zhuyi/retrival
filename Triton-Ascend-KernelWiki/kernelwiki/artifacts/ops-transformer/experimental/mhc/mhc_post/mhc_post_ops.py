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
"""mhc_post operator: Broadcast 1 stream to N streams with scaling.

This module provides PyTorch interface to the AscendC mhc_post kernel.

Usage:
    import mhc_post_ops
    output = mhc_post_ops.mhc_post(input_tensor, h_post_weights)
"""

import logging

import torch
import torch_npu

logger = logging.getLogger(__name__)

try:
    import mhc_post_ext
    _USE_CPP_EXT = True
except ImportError:
    _USE_CPP_EXT = False
    logger.warning("mhc_post_ext not found. Run 'python setup.py build_ext --inplace' to build.")


def mhc_post(x: torch.Tensor, h_post: torch.Tensor) -> torch.Tensor:
    """Broadcast 1 stream to N streams with per-stream scaling.

    Mathematical operation:
        out[b*N + n, s, d] = x[b, s, d] * h_post[n]

    Equivalent einsum:
        out = torch.einsum('bsd,n->bnsd', x, h_post).reshape(B*N, S, D)

    Args:
        x: Input tensor [batch, seq_len, dim]
        h_post: Weight tensor [num_streams]

    Returns:
        Output tensor [batch * num_streams, seq_len, dim]
    """
    if not _USE_CPP_EXT:
        raise RuntimeError("mhc_post_ext not available. Build with setup.py first.")

    return mhc_post_ext.forward(x.contiguous(), h_post.contiguous())


def mhc_post_einsum(x: torch.Tensor, h_post: torch.Tensor) -> torch.Tensor:
    """Reference implementation using torch.einsum."""
    batch = x.size(0)
    num_streams = h_post.size(0)
    if num_streams == 0:
        raise ValueError("num_streams must be > 0, got 0")
    return torch.einsum('bsd,n->bnsd', x, h_post).reshape(batch * num_streams, -1, x.size(-1))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    torch.npu.set_device(0)

    batch, seq_len, dim, num_streams = 2, 16, 32, 4
    x = torch.randn(batch, seq_len, dim, dtype=torch.float32).npu()
    h = torch.randn(num_streams, dtype=torch.float32).npu()

    out_npu = mhc_post(x, h)
    out_ref = mhc_post_einsum(x, h)

    logger.info("Input: %s, h_post: %s", x.shape, h.shape)
    logger.info("Output: %s", out_npu.shape)
    logger.info("Match: %s", torch.allclose(out_npu, out_ref, atol=1e-5))
    logger.info("Max diff: %.2e", (out_npu - out_ref).abs().max().item())
