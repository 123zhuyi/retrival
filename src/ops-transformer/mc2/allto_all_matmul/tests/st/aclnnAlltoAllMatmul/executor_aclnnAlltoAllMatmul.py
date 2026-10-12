# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch.distributed as dist
try:
   import torch_npu
except ImportError:
   pass

import ctypes
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray

@register("function_aclnn_alltoall_matmul")
class AllToAllMatmul(BaseApi):
    def __init__(self, task_result):
        super(AllToAllMatmul, self).__init__(task_result)
        self.dist_task_info = task_result.dist_task_info
        self.rank = self.dist_task_info.rank
        self.world_size = self.dist_task_info.world_size

    def manual_all_to_all(self, x_splits_from_ranks, x_splits_to_ranks):
        for target_rank in range(self.world_size):
            if target_rank == self.rank:
                # 发给自己
                x_splits_from_ranks[self.rank].copy_(x_splits_to_ranks[target_rank])
                # 从别的进程获取我要的张量
                for src_rank in range(self.world_size):  
                    if src_rank != self.rank:
                        dist.recv(
                            tensor=x_splits_from_ranks[src_rank],  # 接收数据的空张量（对应第src_rank个rank）
                            src=src_rank,                              # 源rank
                            tag=src_rank * 1000 + self.rank                  # 匹配发送方的标签（源rank*1000+当前rank）
                        )
            else:
                # 发给其他rank
                dist.send(
                    tensor=x_splits_to_ranks[target_rank],  # 要发送的张量（给第target_rank个rank）
                    dst=target_rank,                            # 目标rank
                    tag=self.rank * 1000 + target_rank                # 通信标签（避免数据混淆）
                )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        world_size = self.dist_task_info.world_size
        x = input_data.kwargs['x']
        weight = input_data.kwargs['weight']
        M = x.shape[0]
        m = int(M / world_size)
        k = x.shape[1]
        bias = input_data.kwargs['bias']
        hasBias = input_data.kwargs['hasBias']

        if self.name == "cpu":
            '''
            ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ CPU  真值 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            '''
            # ========================= alltoall =========================
            x = x.cpu().to(torch.float32)
            weight = weight.cpu().to(torch.float32)
            bias = bias.cpu().to(torch.float32)
            x_splits_from_ranks = [torch.empty((m, k), dtype=x.dtype, device=x.device) 
                                                        for _ in range(world_size)]
            self.manual_all_to_all(x_splits_from_ranks, [x[m * i: m * (i + 1)] for i in range(world_size)])
            A = torch.cat(x_splits_from_ranks, dim = 1)

            # ========================= matmul =========================
            output = torch.matmul(A, weight)
            if hasBias:
                output = output + bias.to(A.dtype)
            return output, A
            '''
            ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ CPU  真值 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            '''

        if self.dist_task_info.is_bm:
            '''
            ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ NPU 小算子级联标杆 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            '''
            # ========================= alltoall =========================
            x_splits_from_ranks = [torch.empty((m, k), dtype=x.dtype, device=x.device) 
                                                        for _ in range(world_size)]  # 承接别的rank
            # 切分送至别的rank
            dist.all_to_all(x_splits_from_ranks, [x[m * i: m * (i + 1)] for i in range(world_size)])
            # 还原隐层
            A = torch.cat(x_splits_from_ranks, dim = 1)

            # ========================= matmul =========================
            output = torch.matmul(A, weight)
            if hasBias:
                output = output + bias.to(A.dtype)
            return output, A
            '''
            ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ NPU 小算子级联标杆 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            '''
        else:
            pass

    def init_by_input_data(self, input_data: InputDataset):
        # 通信域配置
        if self.device == 'pyaclnn' and dist.is_available():
            from torch.distributed.distributed_c10d import _get_default_group
            default_pg = _get_default_group()
            input_data.kwargs['group'] = default_pg


'''
    =================================================================================
    ==============================ACLNN输入特殊处理===================================
    =================================================================================
'''

@register("pyaclnn_aclnn_alltoall_matmul")
class AclnnAllToAllMatmul(AclnnBaseApi):
    def __init__(self, task_result, backend):
        super(AclnnAllToAllMatmul, self).__init__(task_result, backend)
        self.dist_task_info = task_result.dist_task_info

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output


    def init_by_input_data(self, input_data: InputDataset):
        # 处理group
        rank_id = self.dist_task_info.rank
        self.group = input_data.kwargs['group']
        input_data.kwargs['group'] = self.group._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id)
        hasBias = input_data.kwargs['hasBias']
        input_data.kwargs.pop('hasBias')
        if input_data.kwargs['transposeX2']:
            input_data.kwargs['weight'] = input_data.kwargs['weight'].T.clone()
        # =======================================================================
        # =============================== 转化后 ================================
        # =======================================================================
        input_args, output_packages = super().init_by_input_data(input_data)

        if not hasBias:
            input_args[2] = self.get_null_tensor_pte()
        # alltoAllAxesOptional置空
        AclIntArrayPtr = ctypes.POINTER(AclIntArray)
        null_void_ptr = ctypes.c_void_p(None)
        null_array_ptr = ctypes.cast(null_void_ptr, AclIntArrayPtr)
        input_args[3] = null_array_ptr

        return input_args, output_packages

    def get_null_tensor_pte(self):
        AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)  # 把这个空指针类型转换为tensor指针类型
        return null_tensor_ptr