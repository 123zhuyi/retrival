#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
import torch_npu
import torch.nn.functional as F
import ascend_ops
import pytest


def test_layer_norm_interface_exist():
    """
    Test that the 'ascend_ops.layer_norm' operator is present in torch.ops.
    """
    print(torch.ops.ascend_ops.layer_norm)
    assert hasattr(torch.ops.ascend_ops, "layer_norm"), \
        "The 'layer_norm' operator is not registered in the 'torch.ops.ascend_ops' namespace."


# Test cases from original cann-samples
# Format: (rows, hiddenSize)
SHAPES = [
    (4096, 8192),
    (128, 768),
    (64, 100),
    (32, 13),
    (1, 256),
    (256, 1),
    (16, 9),
    (32, 16384),
    (8, 12288),
]

DTYPES = [
    torch.float32,
    torch.float16,
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("rows, hidden_size", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_layer_norm_operator(rows, hidden_size, dtype):
    """
    Test the functionality of the layer_norm operator.
    
    LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
    """
    eps = 1e-6
    
    x = torch.randn(rows, hidden_size, dtype=dtype)
    gamma = torch.randn(hidden_size, dtype=dtype)
    beta = torch.randn(hidden_size, dtype=dtype)
    
    expected = F.layer_norm(x, [hidden_size], gamma, beta, eps=eps)
    
    x_npu = x.npu()
    gamma_npu = gamma.npu()
    beta_npu = beta.npu()
    result_npu = torch.ops.ascend_ops.layer_norm(x_npu, gamma_npu, beta_npu, eps)
    result = result_npu.cpu()
    
    rtol = 5e-2 if dtype == torch.float16 else 1e-3
    atol = 1e-2 if dtype == torch.float16 else 1e-4
    
    assert torch.allclose(result, expected, rtol=rtol, atol=atol), \
        f"LayerNorm failed for shape ({rows}, {hidden_size}), dtype {dtype}. " \
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"
    
    print(f"Test passed: shape=({rows}, {hidden_size}), dtype={dtype}")


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_layer_norm_small():
    """Test with small dimensions."""
    eps = 1e-6
    x = torch.randn(2, 4, dtype=torch.float32)
    gamma = torch.randn(4, dtype=torch.float32)
    beta = torch.randn(4, dtype=torch.float32)
    
    expected = F.layer_norm(x, [4], gamma, beta, eps=eps)
    
    x_npu = x.npu()
    gamma_npu = gamma.npu()
    beta_npu = beta.npu()
    result_npu = torch.ops.ascend_ops.layer_norm(x_npu, gamma_npu, beta_npu, eps)
    result = result_npu.cpu()
    
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-4), \
        f"LayerNorm failed for small test. Max diff: {torch.max(torch.abs(result - expected)):.6f}"


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
def test_layer_norm_bert_base():
    """Test with BERT-base dimensions."""
    eps = 1e-6
    batch_size = 32
    seq_len = 128
    hidden_size = 768
    
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    x = x.view(batch_size * seq_len, hidden_size)
    gamma = torch.randn(hidden_size, dtype=torch.float32)
    beta = torch.randn(hidden_size, dtype=torch.float32)
    
    expected = F.layer_norm(x, [hidden_size], gamma, beta, eps=eps)
    
    x_npu = x.npu()
    gamma_npu = gamma.npu()
    beta_npu = beta.npu()
    result_npu = torch.ops.ascend_ops.layer_norm(x_npu, gamma_npu, beta_npu, eps)
    result = result_npu.cpu()
    
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-4), \
        f"LayerNorm failed for BERT-base. Max diff: {torch.max(torch.abs(result - expected)):.6f}"
