#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import test
import torch
import torch_npu
import check_valid_param
import pytest


def test_fa_no_quant(params):
    batch_size, q_head_num, kv_head_num, q_seq, kv_seq, head_dim, dtype, in_layout, \
    act_seq_len, act_seq_len_kv, scaled_value, rope = params
    
    # bnsd
    if in_layout == "BNSD":
        query_cpu = torch.randn(batch_size, q_head_num, q_seq, head_dim + rope).to(dtype)
        key_cpu = torch.randn(batch_size, kv_head_num, kv_seq, head_dim + rope).to(dtype)
        value_cpu = torch.randn(batch_size, kv_head_num, kv_seq, head_dim).to(dtype)
        enable_gqa = q_head_num != kv_head_num
        expect = torch.nn.functional.scaled_dot_product_attention(query_cpu, key_cpu, value_cpu, scale=scaled_value, enable_gqa=enable_gqa)
    else:
        query_cpu = torch.randn(batch_size, q_seq, q_head_num, head_dim + rope).to(dtype)
        key_cpu = torch.randn(batch_size, kv_seq, kv_head_num, head_dim + rope).to(dtype)
        value_cpu = torch.randn(batch_size, kv_seq, kv_head_num, head_dim).to(dtype)
        enable_gqa = q_head_num != kv_head_num
        query_bnsd = query_cpu.transpose(1, 2)
        key_bnsd = key_cpu.transpose(1, 2)
        value_bnsd = value_cpu.transpose(1, 2)
        expect = torch.nn.functional.scaled_dot_product_attention(query_bnsd, key_bnsd, value_bnsd, scale=scaled_value, enable_gqa=enable_gqa)
        expect = expect.transpose(1, 2)
    if rope > 0:
        q_tensor = query_cpu[:, :, :, :head_dim].npu()
        q_rope = query_cpu[:, :, :, head_dim:].npu()

        k_tensor = key_cpu[:, :, :, :head_dim].npu()
        k_rope = key_cpu[:, :, :, head_dim:].npu()
    else:
        q_tensor = query_cpu.npu()
        k_tensor = key_cpu.npu()
        q_rope = None
        k_rope = None
    v_tensor = value_cpu.npu()
    # 调用FIA算子
    result, _ = torch_npu.npu_fused_infer_attention_score(q_tensor, k_tensor, v_tensor,
            num_heads=q_head_num, num_key_value_heads=kv_head_num,
            input_layout=in_layout, scale=scaled_value, query_rope=q_rope, key_rope=k_rope
            )
    torch.npu.synchronize()
    result = result.cpu()
    return expect, result