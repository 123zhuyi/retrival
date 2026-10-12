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

@register("execute_aclnnMatmulAllReduceV2")
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
        x3_chunk = input_data.kwargs['x3']
        bias_chunk = input_data.kwargs['bias']
        if input_chunk.shape != torch.Size([]):
            m_size = input_chunk.shape[0] // 8
            if m_size != 0:
                start = self.dist_task_info.rank * m_size
                end = start + m_size
                input_chunk = input_chunk[start:end,:]

            if weight_chunk.shape[0] != input_chunk.shape[-1] and weight_chunk.shape[1] == input_chunk.shape[-1]:
                weight_chunk = weight_chunk.transpose(0, 1)

            if self.task_result.case_config.id % 4 == 1:
                bias_chunk = None
                x3_chunk = None

        if self.name == "cpu":
            input_chunk = input_chunk.cpu().to(torch.float32)
            weight_chunk = weight_chunk.cpu().to(torch.float32).contiguous()
            output = torch.matmul(input_chunk, weight_chunk)
            if bias_chunk != None:
                bias_chunk = bias_chunk.to(torch.float32).cpu()
                output = torch.add(output, bias_chunk)
            if x3_chunk != None:
                x3_chunk = x3_chunk.to(torch.float32).cpu()
                output = torch.add(output, x3_chunk)
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            return output
        if self.dist_task_info.is_bm:
            output = torch.matmul(input_chunk, weight_chunk)
            if bias_chunk != None:
                output = torch.add(output, bias_chunk)
            if x3_chunk != None:
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
                                                           reduce_op="sum"
                                                           )
            return output
