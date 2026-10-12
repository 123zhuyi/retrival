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

import torch
import torch_npu
import ascend_ops

supported_dtypes = {torch.bfloat16}
E = 3
M = 64
K = 32
N = 32
PERGROUP = M

def generate_group_list_tensor(E, M, PERGROUP):
    group_list = torch.zeros(E, dtype=torch.int64)
    total_group = PERGROUP
    
    for i in range(E):
        if total_group <= M:
            group_list[i] = PERGROUP
            total_group += PERGROUP
        else:
            group_list[i] = PERGROUP - (total_group - M)
            break
    
    return group_list

group_list = generate_group_list_tensor(E, M, PERGROUP)

EPS = 0.001

for data_type in supported_dtypes:
    print(f"DataType = <{data_type}>")
    
    x_list = []
    x_cpu = torch.rand(M, K, dtype=data_type)
    x_list.append(x_cpu)

    weight_list = []
    weight_cpu = torch.rand(K, N, dtype=data_type)
    weight_list.append(weight_cpu)
    
    # 将数据移动到NPU
    x_list_npu = [x_i.npu() for x_i in x_list]
    weight_list_npu = [weight_i.npu() for weight_i in weight_list]
    group_list_npu = group_list.npu()
    
    # 调用groupedmatmul，提供所有必需的参数
    try:
        npu_result = ascend_ops.ops.groupedmatmul(
            x_list_npu,           # Tensor[] x
            weight_list_npu,      # Tensor[] weight
            None,                 # Tensor[]? bias (可选)
            None,                 # Tensor[]? scale (可选)
            None,                 # Tensor[]? offset (可选)
            None,                 # Tensor[]? antiquantScale (可选)
            None,                 # Tensor[]? antiquantOffset (可选)
            group_list_npu,       # Tensor? groupList
            None,                 # Tensor[]? perTokenScale (可选)
            3,                    # int splitItem
            0,                    # int groupType
            1,                    # int groupListType
            0,                    # int actType
            None                  # int[]? tuningConfigOptional (可选)
        ).cpu()
        
        print(f"Result shape: {npu_result.shape} \n",npu_result)
        
    except Exception as e:
        print(f"Error calling groupedmatmul: {e}")
    
    torch_result = None
    try:
        x = x_list[0].to(torch.float32)
        weight = weight_list[0].to(torch.float32)
        
        group_size = M // E
        remainder = M % E
        split_sizes = [group_size] * E
        if remainder > 0:
            split_sizes[-1] += remainder
        
        x_splits = torch.split(x, split_sizes, dim=0)
        
        split_results = []
        for x_split in x_splits:
            split_matmul = torch.matmul(x_split, weight)
            split_results.append(split_matmul)
        
        torch_result = torch.cat(split_results, dim=0).to(data_type)
        print(f"PyTorch Result shape: {torch_result.shape}")
        print(f"分组大小: {split_sizes}, 各分组结果形状: {[r.shape for r in split_results]}")
    except Exception as e:
        print(f"Error calculating PyTorch grouped matmul: {e}")
        continue

    try:
        if npu_result.shape != torch_result.shape:
            print(f"Shape mismatch! NPU: {npu_result.shape}, PyTorch: {torch_result.shape}")
            print("精度对比失败")
            continue
        
        npu_float = npu_result.to(torch.float32)
        torch_float = torch_result.to(torch.float32)
        
        abs_diff = torch.abs(npu_float - torch_float)
        bad_indices = torch.where(abs_diff > EPS)
        bad_values = abs_diff[bad_indices]
        
        if len(bad_values) > 0:
            print(f"\n精度对比失败！发现 {len(bad_values)} 个点位差值超过 {EPS}:")
            for idx in range(min(len(bad_values), 10)):
                i, j = bad_indices[0][idx].item(), bad_indices[1][idx].item()
                diff = bad_values[idx].item()
                npu_val = npu_float[i, j].item()
                torch_val = torch_float[i, j].item()
                print(f"点位({i}, {j}): NPU={npu_val:.6f}, PyTorch={torch_val:.6f}, 差值={diff:.6f}")
            print("精度对比失败！")
        else:
            max_diff = torch.max(abs_diff).item()
            print(f"\n精度对比通过！最大绝对误差: {max_diff:.6f} (阈值={EPS})")
    except Exception as e:
        print(f"Error comparing results: {e}")