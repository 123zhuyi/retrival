# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch


def type_to_byte(dtype):
    if dtype == torch.bfloat16:
        return 2 
    elif dtype == torch.float16:
        return 2 
    elif dtype == torch.int8:
        return 1
    else:
        raise NotImplementedError


def calc_num_ops(q, k, v, data_layout):
    if data_layout == "TND":
        bsz, n_heads, q_headdim = q.shape
        seqlen, n_kv_heads, k_headdim = k.shape
        seqlen, n_kv_heads, v_headdim = v.shape    

        n_ops = bsz * seqlen * n_heads * q_headdim + bsz * seqlen * n_heads * v_headdim
        n_ops = 2 * n_ops
        return n_ops
    elif data_layout == "BNSD":
        bsz, n_heads, _, q_headdim = q.shape
        bsz, n_kv_heads, seqlen, k_headdim = k.shape
        bsz, n_kv_heads, seqlen, v_headdim = v.shape    

        n_ops = bsz * seqlen * n_heads * q_headdim + bsz * seqlen * n_heads * v_headdim
        n_ops = 2 * n_ops
        return n_ops        
    elif data_layout == "BSH":
        # q_hiddendim is equal to n_heads x headdim
        bsz, _, q_hiddendim = q.shape
        bsz, seqlen, _ = k.shape
        bsz, seqlen, _ = v.shape    

        n_ops = bsz * seqlen * q_hiddendim + bsz * seqlen * q_hiddendim
        n_ops = 2 * n_ops
        return n_ops        
    else:
        raise NotImplementedError


def calc_mem_size(q, k, v, data_layout):
    if data_layout == "TND":
        bsz, n_heads, q_headdim = q.shape
        seqlen, n_kv_heads, k_headdim = k.shape
        seqlen, n_kv_heads, v_headdim = v.shape    

        mem_size = bsz * n_heads * q_headdim * type_to_byte(q.dtype)
        mem_size += seqlen * n_kv_heads * k_headdim * type_to_byte(k.dtype)
        mem_size += seqlen * n_kv_heads * v_headdim * type_to_byte(v.dtype)
        
        return mem_size
    elif data_layout == "BNSD":
        bsz, n_heads, _, q_headdim = q.shape
        bsz, n_kv_heads, seqlen, k_headdim = k.shape
        bsz, n_kv_heads, seqlen, v_headdim = v.shape 

        mem_size = bsz * n_heads * q_headdim * type_to_byte(q.dtype)
        mem_size += bsz * seqlen * n_kv_heads * k_headdim * type_to_byte(k.dtype)
        mem_size += bsz * seqlen * n_kv_heads * v_headdim * type_to_byte(v.dtype)

        return mem_size
    elif data_layout == "BSH":
        # q_hiddendim is equal to n_heads x headdim
        bsz, _, q_hiddendim = q.shape
        bsz, seqlen, kv_hiddendim = k.shape
        bsz, seqlen, kv_hiddendim = v.shape    

        mem_size = bsz * q_hiddendim * type_to_byte(q.dtype)
        mem_size += bsz * seqlen * kv_hiddendim * type_to_byte(k.dtype)
        mem_size += bsz * seqlen * kv_hiddendim * type_to_byte(v.dtype)
        
        return mem_size    
    else:
        raise NotImplementedError


def calc_cube_throughput(n_ops, elapsed_time):
    # elapsed_time in ms, convert it to s
    elapsed_time = elapsed_time * 1e-3
    return n_ops / elapsed_time * 1e-12 # TOPS/s


def calc_hbm_throughput(mem_size, elapsed_time):
    # elapsed_time in ms, convert it to s
    elapsed_time = elapsed_time * 1e-3
    return mem_size / elapsed_time * 1e-12 # TB/s


def softmax(x, use_pytorch=False):
    if use_pytorch:
        denom = torch.sum(torch.exp(x.to(torch.float32)), dim=-1, keepdim=True)
        lse = torch.log(denom)
        out = torch.nn.functional.softmax(x, dim=-1)
        return out, lse
    else:
        max_scores, _ = torch.max(x, dim=-1, keepdim=True)
        lse = torch.log(torch.sum(torch.exp(x.to(torch.float32) - max_scores), dim=-1, keepdim=True)) + max_scores
        out = torch.exp(x - lse)
        return out.to(x.dtype), lse
