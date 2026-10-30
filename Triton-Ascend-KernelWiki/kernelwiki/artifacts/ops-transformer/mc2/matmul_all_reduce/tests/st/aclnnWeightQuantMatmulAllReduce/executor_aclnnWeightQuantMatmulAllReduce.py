#!/usr/bin/env python3
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

import math
import os.path
import numpy as np
from itertools import product
import time
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
try:
   import torch_npu
except ImportError:
   pass

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

@register("execute_aclnnWeightQuantMatmulAllReduce")
class DistFunctionApi(BaseApi):
    def __init__(self, task_result):
        super(DistFunctionApi, self).__init__(task_result)
        self.dist_task_info = task_result.dist_task_info

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        rank_id = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size
        input_chunk = input_data.kwargs['x1']

        if input_chunk.shape == torch.Size([]):
            if self.name == "cpu" or self.dist_task_info.is_bm :
                return torch.tensor([])

        weight_chunk = input_data.kwargs['x2']
        if input_chunk.shape != torch.Size([]):
            m_size = input_chunk.shape[0] // 8
            if m_size != 0:
                start = rank_id * m_size
                end = start + m_size
                input_chunk = input_chunk[start:end,:]

        if input_chunk.shape == torch.Size([2147483647,1]):
            if self.name == "cpu" or self.dist_task_info.is_bm :
                return torch.zeros(input_chunk.shape,dtype=input_chunk.dtype)

        if len(input_chunk.shape)==3:
            input_golden = input_chunk.reshape(-1, input_chunk.shape[-1])
        else:
            input_golden = input_chunk

        CASES = list(product((False, True), repeat=4))
        optional_input = input_data.kwargs['optional_input']
        is_transpose = input_data.kwargs['isTranspose']

        is_antiquant_offset, is_bias, is_x3 , is_nz = CASES[optional_input]

        is_nz = False
        if is_nz:
            weight_chunk = torch_npu.npu_format_cast(weight_chunk, 29)

        if is_transpose:
            weight_chunk = weight_chunk.transpose(0, 1)


        antiquant_scale_chunk = input_data.kwargs['antiquantScale']
        antiquantGroupSize = input_data.kwargs['antiquantGroupSize']
        x3_chunk = input_data.kwargs['x3'] if is_x3 else None
        bias_chunk = input_data.kwargs['bias'] if is_bias else None
        antiquant_offset_chunk = input_data.kwargs['antiquantOffset'] if is_antiquant_offset else None

        if antiquant_scale_chunk.shape[-1] != weight_chunk.shape[-1] and antiquant_scale_chunk.shape != torch.Size([1]):
            antiquant_scale_chunk = antiquant_scale_chunk.transpose(0, 1)
            if is_antiquant_offset:
                antiquant_offset_chunk = antiquant_offset_chunk.transpose(0, 1)

        # per-group场景
        # if antiquantGroupSize > 0:
        #     tmp = ( antiquant_scale_chunk.shape[0] + antiquantGroupSize - 1 )// antiquantGroupSize
        #     antiquant_scale_chunk = antiquant_scale_chunk [0:tmp,:]
        #     antiquant_offset_chunk = antiquant_offset_chunk [0:tmp,:] if is_antiquant_offset else None

        if self.name == "cpu":
            input_chunk = input_chunk.to(torch.float32).cpu()
            weight_chunk = weight_chunk.to(torch.float32).cpu().contiguous()
            antiquant_scale = antiquant_scale_chunk.cpu().to(torch.float32)

            if antiquantGroupSize != 0:  # per-group场景
                for i in range(input_chunk.shape[-1]):
                    group_size_dim = i // antiquantGroupSize
                    if is_antiquant_offset:
                        antiquant_offset = antiquant_offset_chunk.cpu().to(torch.float32)
                        weight_chunk[i, :] = (weight_chunk[i, :] + antiquant_offset[group_size_dim, :]) * antiquant_scale[group_size_dim, :]
                    else:
                        weight_chunk[i, :] = weight_chunk[i, :] * antiquant_scale[group_size_dim, :]
            else:  # per-tensor/per-channel场景
                if is_antiquant_offset:
                    antiquant_offset = antiquant_offset_chunk.cpu().to(torch.float32)
                    weight_chunk = torch.add(weight_chunk, antiquant_offset)
                weight_chunk = torch.mul(weight_chunk, antiquant_scale)
            output = torch.matmul(input_chunk, weight_chunk)
            if bias_chunk != None:
                bias_chunk = bias_chunk.cpu()
                output = torch.add(output, bias_chunk)
            if x3_chunk != None:
                x3_chunk = x3_chunk.to(torch.float32).cpu()
                output = torch.add(output, x3_chunk)
            # 通信部分
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            return output
        if self.dist_task_info.is_bm:
            if bias_chunk != None and bias_chunk.dtype == torch.bfloat16:
                bias_chunk = bias_chunk.to(torch.float32)
            output = torch_npu.npu_weight_quant_batchmatmul(x=input_golden, weight=weight_chunk, bias=bias_chunk,
                                                    antiquant_scale=antiquant_scale_chunk, antiquant_offset=antiquant_offset_chunk,
                                                    antiquant_group_size=antiquantGroupSize)
            if x3_chunk != None:
                output = output.reshape(x3_chunk.shape)
                output = torch.add(output, x3_chunk)
            dist.all_reduce(output, op=ReduceOp.SUM)
            return output
        else:
            if dist.is_available():
                from torch.distributed.distributed_c10d import _get_default_group
                default_pg = _get_default_group()
                if torch.__version__ > '2.0.1':
                    hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id)
                else:
                    hcomm_info = default_pg.get_hccl_comm_name(rank_id)
            output = torch_npu.npu_mm_all_reduce_base(x1=input_chunk,
                                                            x2=weight_chunk,
                                                            x3=x3_chunk,
                                                            bias=bias_chunk,
                                                            hcom=hcomm_info,
                                                            reduce_op="sum",
                                                            antiquant_scale=antiquant_scale_chunk,
                                                            antiquant_offset=antiquant_offset_chunk,
                                                            antiquant_group_size=antiquantGroupSize
                                                           )
            return output
