#!/usr/bin/python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_npu
from torch_npu.contrib import transfer_to_npu

from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from torch.profiler import profile, schedule
from torch.autograd.profiler import record_function
from torch.cuda.amp import autocast

from einops import rearrange, repeat

os.environ["COMBINED_ENABLE"] = "1"  # 
os.environ["INF_NAN_MODE_ENABLE"] = "1"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
torch.manual_seed(0)
torch.set_printoptions(precision=16)

torch_npu.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.config.allow_internal_format = False
torch.npu.conv.allow_hf32 = False
torch.npu.matmul.allow_hf32 = False 


def get_interleave_matrix(n):
    matrix = torch.zeros(n, n, dtype=torch.bfloat16, device='npu')
    for i in range(0, n, 2):
        matrix[i + 0, i + 1] = 1
        matrix[i + 1, i + 0] = -1
    return matrix


def get_half_matrix(n):
    matrix = torch.zeros(n, n, dtype=torch.bfloat16)
    half = n // 2
    matrix[:half, half:] = torch.eye(half)
    matrix[half:, :half] = -torch.eye(half)
    return matrix.to('npu')


def compose_3matrix(matrix_a, matrix_b, matrix_c):
    total_rows = matrix_a.size(0) + matrix_b.size(0) + matrix_c.size(0)
    total_cols = matrix_a.size(1) + matrix_b.size(1) + matrix_c.size(1)

    result = torch.zeros((total_rows, total_cols), dtype=torch.bfloat16)

    result[:matrix_a.size(0), :matrix_a.size(1)] = matrix_a

    b_row_start = matrix_a.size(0)
    b_col_start = matrix_a.size(1)
    result[b_row_start:b_row_start + matrix_b.size(0),
           b_col_start:b_col_start + matrix_b.size(1)] = matrix_b
    
    c_row_start = matrix_a.size(0) + matrix_b.size(0)
    c_col_start = matrix_a.size(1) + matrix_b.size(1)
    result[c_row_start:c_row_start + matrix_c.size(0),
           c_col_start:c_col_start + matrix_c.size(1)] = matrix_c
    
    return result.to('npu')


# half
def rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# interleave
def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.chunk(2, dim=-1)
    x = torch.cat((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, mode: str) -> torch.Tensor:
    sin = sin.unsqueeze(0).unsqueeze(1)
    cos = cos.unsqueeze(0).unsqueeze(1)
    if mode == 'interleave':
        return (tensor * cos) + (rotate_every_two(tensor) * sin)
    elif mode == 'half':
        return (tensor * cos) + (rotate_half(tensor) * sin)
    else:
        raise NotImplementedError("mode error, only support half or interleave")
    

def apply_3drotary_pos_v3(q, k, freqs_cis, mat, high_precision=None):
    '''
    本方案:rope_matrix, matrix合一
    '''
    sincos_h, sincos_w, sincos_t = freqs_cis

    sin_h, cos_h = sincos_h
    sin_w, cos_w = sincos_w
    sin_t, cos_t = sincos_t

    sin = torch.cat((sin_h, sin_w, sin_t), dim=-1)
    cos = torch.cat((cos_h, cos_w, cos_t), dim=-1)
    
    if high_precision:
        q = q.float() * cos.float() + (q @ mat).float() * sin.float()
        k = k.float() * cos.float() + (k @ mat).float() * sin.float()
        q, k = q.to(torch.bfloat16), k.to(torch.bfloat16)
    else:
        q = q * cos + (q @ mat) * sin
        k = k * cos + (k @ mat) * sin
    
    return q, k


def apply_3drotary_pos_v5(q, k, freqs_cis, mat):
    '''
    本方案:自定义融合算子实现
    '''
    import npu_ops_transformer_ext
    sincos_h, sincos_w, sincos_t = freqs_cis

    sin_h, cos_h = sincos_h
    sin_w, cos_w = sincos_w
    sin_t, cos_t = sincos_t

    sin = torch.cat((sin_h, sin_w, sin_t), dim=-1)
    cos = torch.cat((cos_h, cos_w, cos_t), dim=-1)

    def rope(x, mat, cos, sin):
        x = torch.ops.npu_ops_transformer_ext.rope_matrix(x, mat, sin, cos)
        return x

    q = rope(q, mat, cos, sin)
    k = rope(k, mat, cos, sin)
    return q, k


class ROPE3D(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, q, k, freqs_cis, mat):
        return apply_3drotary_pos_v3(q, k, freqs_cis, mat)
    

def main():
    # init
    b, n, s, d = 1, 24, 28800, 128
    shape_lists_1d = [128]
    shape_lists_2d = [64, 64]
    shape_lists_3d = [44, 44, 40]

    q = torch.randn((b, n, s, d), dtype=torch.bfloat16, device='npu')
    k = torch.randn((b, n, s, d), dtype=torch.bfloat16, device='npu')
    freqs_cis_3d = []
    for shape in shape_lists_3d:
        sincos = torch.randn((s, shape), dtype=torch.bfloat16, device='npu')
        freqs_cis_3d.append([sincos, sincos])

    inter_mat_128 = get_interleave_matrix(128)
    half_mat_44 = get_half_matrix(44)
    half_mat_40 = get_half_matrix(40)
    half_mat_44_44_40 = compose_3matrix(half_mat_44, half_mat_44, half_mat_40)
    
    # test precision
    with record_function("3d rope v5"):
        outq5, outk5 = apply_3drotary_pos_v5(q, k, freqs_cis_3d, half_mat_44_44_40)
        torch.cuda.synchronize()
    
    with record_function("3d rope v3"):
        outq3, outk3 = apply_3drotary_pos_v3(q, k, freqs_cis_3d, half_mat_44_44_40, high_precision=True)
        torch.cuda.synchronize()
    
    # 精度验证
    denominator = torch.maximum(torch.abs(outq3), torch.abs(outq5))
    abs_error = torch.abs(torch.abs(outq3) - torch.abs(outq5))
    error = abs_error / (denominator + 1e-8)
    assert ((torch.minimum(abs_error, error) > 0).sum() == 0)

    
if __name__ == '__main__':
    main()