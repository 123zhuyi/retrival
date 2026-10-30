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
"""mhc_res operator: Stream mixing via learned weight matrix.

This module provides PyTorch interface to the AscendC mhc_res kernel.

Usage:
    import mhc_res_ops
    output = mhc_res_ops.mhc_res(input_tensor, h_res_matrix)
"""

import logging

import torch
import torch_npu

logger = logging.getLogger(__name__)

try:
    import mhc_res_ext
    _USE_CPP_EXT = True
except ImportError:
    _USE_CPP_EXT = False
    logger.warning("mhc_res_ext not found. Run 'python setup.py build_ext --inplace' to build.")


def mhc_res(x: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
    """Mix streams via learned weight matrix (residual path).

    Mathematical operation:
        out[b*N + t, s, d] = sum_r(h_res[r, t] * x[b*N + r, s, d])

    Equivalent einsum:
        x_4d = x.view(batch, num_streams, seq_len, dim)
        out = torch.einsum('brsd,rt->btsd', x_4d, h_res).reshape(B*N, S, D)

    Args:
        x: Input tensor [batch * num_streams, seq_len, dim]
        h_res: Weight matrix [num_streams, num_streams]

    Returns:
        Output tensor [batch * num_streams, seq_len, dim]
    """
    if not _USE_CPP_EXT:
        raise RuntimeError("mhc_res_ext not available. Build with setup.py first.")

    return mhc_res_ext.forward(x.contiguous(), h_res.contiguous())


def mhc_res_einsum(x: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
    """Reference implementation using torch.einsum."""
    num_streams = h_res.size(0)
    if num_streams == 0:
        raise ValueError("num_streams must be > 0, got 0")
    batch = x.size(0) // num_streams
    seq_len = x.size(1)
    dim = x.size(2)
    x_4d = x.view(batch, num_streams, seq_len, dim)
    return torch.einsum('brsd,rt->btsd', x_4d, h_res).reshape(batch * num_streams, seq_len, dim)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    torch.npu.set_device(0)

    batch, seq_len, dim, num_streams = 2, 16, 32, 4
    x = torch.randn(batch * num_streams, seq_len, dim, dtype=torch.float32).npu()
    h = torch.randn(num_streams, num_streams, dtype=torch.float32).npu()

    out_npu = mhc_res(x, h)
    out_ref = mhc_res_einsum(x, h)

    logger.info("Input: %s, h_res: %s", x.shape, h.shape)
    logger.info("Output: %s", out_npu.shape)
    logger.info("Match: %s", torch.allclose(out_npu, out_ref, atol=1e-5))
    logger.info("Max diff: %.2e", (out_npu - out_ref).abs().max().item())
