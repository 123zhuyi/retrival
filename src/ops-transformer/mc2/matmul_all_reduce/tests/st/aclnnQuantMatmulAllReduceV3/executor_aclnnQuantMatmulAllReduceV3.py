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

@register("execute_aclnnQuantMatmulAllReduceV3")
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

        dequant_scale_chunk = input_data.kwargs['dequantScale']
        output_dtype = torch.bfloat16 if dequant_scale_chunk.dtype == torch.bfloat16 else torch.float16

        if input_chunk.shape == torch.Size([2147483647,1]):
            if self.name == "cpu" or self.dist_task_info.is_bm :
                return torch.zeros(input_chunk.shape,dtype=output_dtype)

        use_alltoall = 1

        if len(input_chunk.shape)==3:
            input_golden = input_chunk.reshape(-1, input_chunk.shape[-1])
        else:
            input_golden = input_chunk

        if len(dequant_scale_chunk.shape)==2:
            dequant_scale_golden = dequant_scale_chunk.reshape(dequant_scale_chunk.shape[-1])
        else:
            dequant_scale_golden = dequant_scale_chunk

        CASES = list(product((False, True), repeat=5))
        optional_input = input_data.kwargs['optional_input']

        is_pertoken, is_commquant, is_bias, is_x3 , is_nz = CASES[optional_input]

        is_nz = False
        if is_nz:
            weight_chunk = torch_npu.npu_format_cast(weight_chunk, 29)

        if weight_chunk.shape != torch.Size([]) and weight_chunk.shape[0] != input_chunk.shape[-1] and weight_chunk.shape[1] == input_chunk.shape[-1]:
            weight_chunk = weight_chunk.transpose(0, 1)

        output_dtype = torch.bfloat16 if dequant_scale_chunk.dtype == torch.bfloat16 else torch.float16

        x3_chunk = input_data.kwargs['x3'] if is_x3 else None
        bias_chunk = input_data.kwargs['bias'] if is_bias else None
        pertoken_scale_chunk = input_data.kwargs['pertokenScale'] if is_pertoken else None

        if not is_pertoken and dequant_scale_chunk.dtype == torch.float32 and self.name != "cpu":
            dequant_scale_golden = torch_npu.npu_trans_quant_param(dequant_scale_golden)
            dequant_scale_chunk = torch_npu.npu_trans_quant_param(dequant_scale_chunk)  # torch.int64

        commm_quant_scale_1 = input_data.kwargs['commQuantScale1'] if is_commquant else None
        comm_scale_shape_1 =  commm_quant_scale_1.shape if is_commquant else None

        commm_quant_scale_2 = input_data.kwargs['commQuantScale2'] if is_commquant else None
        comm_scale_shape_2 =  commm_quant_scale_2.shape if is_commquant else None

        save_path = f"./comm_path/{self.task_result.case_config.id}/"
        # 获取comm_quant_scale_1 (由cpu预先计算得出)
        comm_path_1 = f"{save_path}/fp32_comm_quant_scale_1_0.bin"
        comm_path_2 = f"{save_path}/fp32_comm_quant_scale_2_0.bin"
        comm_scale_dtype_1 = output_dtype
        comm_scale_dtype_2 = output_dtype
        if os.path.exists(comm_path_1) and os.path.exists(comm_path_2) and is_commquant:
            tmp_cquant_scale_1 = torch.tensor(
                np.fromfile(comm_path_1, dtype=np.float32).reshape(comm_scale_shape_1)).to(comm_scale_dtype_1)
            commm_quant_scale_1 = tmp_cquant_scale_1 if self.name == "cpu" else tmp_cquant_scale_1.npu()
            tmp_cquant_scale_2 = torch.tensor(
                np.fromfile(comm_path_2, dtype=np.float32).reshape(comm_scale_shape_2)).to(comm_scale_dtype_2)
            commm_quant_scale_2 = tmp_cquant_scale_2 if self.name == "cpu" else tmp_cquant_scale_2.npu()
        else:
            commm_quant_scale_1 = None
            commm_quant_scale_2 = None

        diff_tensor = None
        all_to_all_out = None
        all_gather_out = None
        # 低bit通信场景需要预先计算脏数据tensor、alltoall接收tensor、allgather接收tensor
        if is_commquant and input_chunk.shape != torch.Size([]):
            # 预先创建脏数据tensor--output的M不能被worls_size整除场景
            if len(input_chunk.shape)==3:
                M = input_chunk.shape[0] * input_chunk.shape[1]
            else:
                M = input_chunk.shape[0]
            N = weight_chunk.shape[1]
            final_M = math.ceil(M / world_size) * world_size
            diff_tensor_shape = [int(final_M - M), int(N)]
            diff_tensor = torch.zeros(diff_tensor_shape) if self.name == "cpu" else torch.zeros(diff_tensor_shape).npu()
            # 预先创建alltoall接收tensor
            alltoall_allgather_shape = [int(world_size), int(final_M / world_size), int(N)]
            tmp_alltoall_out = torch.zeros(alltoall_allgather_shape, dtype=torch.int8)
            all_to_all_out = tmp_alltoall_out if self.name == "cpu" else tmp_alltoall_out.npu()
            # 预先创建allgather接收tensor
            tmp_allgather_out = torch.zeros(alltoall_allgather_shape, dtype=torch.int8)
            all_gather_out = tmp_allgather_out if self.name == "cpu" else tmp_allgather_out.npu()

        if self.name == "cpu":
            input_chunk = input_chunk.to(torch.int32).cpu()
            weight_chunk = weight_chunk.to(torch.int32).cpu().contiguous()
            output = torch.matmul(input_chunk, weight_chunk)
            if bias_chunk != None:
                bias_chunk = bias_chunk.cpu()
                output = torch.add(output, bias_chunk)
            if pertoken_scale_chunk != None:
                output_origial_shape = output.shape
                if len(output_origial_shape) == 3:
                    output = output.reshape(-1, output.shape[-1])
                pertoekn_scale = torch.unsqueeze(pertoken_scale_chunk.cpu(), dim=1)
                output = (output * pertoekn_scale).reshape(output_origial_shape)
            dequant_scale_cpu = dequant_scale_chunk.to(torch.float32).cpu()
            output = torch.mul(output, dequant_scale_cpu)
            if x3_chunk != None:
                output = output.reshape(x3_chunk.shape)
                x3_chunk = x3_chunk.to(torch.float32).cpu()
                output = torch.add(output, x3_chunk)
            # 通信部分
            if not is_commquant:
                dist.all_reduce(output, op=dist.ReduceOp.SUM)
            else:
                comm_quant_scale_1 = commm_quant_scale_1.to(
                    torch.float32).cpu() if commm_quant_scale_1 is not None else None
                comm_quant_scale_2 = commm_quant_scale_2.to(
                    torch.float32).cpu() if commm_quant_scale_2 is not None else None

                output = hcom_quant_cpu(output, comm_quant_scale_1, comm_quant_scale_2, world_size,
                                            use_alltoall, save_path, [], rank_id,
                                            out_dtype=output_dtype)
            return output
        if self.dist_task_info.is_bm:
            output = torch_npu.npu_quant_matmul(x1=input_golden, x2=weight_chunk, scale=dequant_scale_golden,
                                                    bias=bias_chunk, pertoken_scale=pertoken_scale_chunk,
                                                    output_dtype=output_dtype)
            if x3_chunk != None:
                output = output.reshape(x3_chunk.shape)
                output = torch.add(output, x3_chunk)
            if not is_commquant:
                dist.all_reduce(output, op=ReduceOp.SUM)
            else:
                output = hcom_quant(output, commm_quant_scale_1, commm_quant_scale_2, diff_tensor,
                                            all_to_all_out, all_gather_out, world_size, use_alltoall)
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
                                                            dequant_scale=dequant_scale_chunk,
                                                            pertoken_scale=pertoken_scale_chunk,
                                                            comm_quant_scale_1=commm_quant_scale_1,
                                                            comm_quant_scale_2=commm_quant_scale_2
                                                           )
            return output

def hcom_quant_cpu(mm_output, comm_quant_scale_1, comm_quant_scale_2, world_size, is_alltoall, save_path=None,
                comm_quant_list=[], rank=0, out_dtype=torch.float16):
    original_shape = mm_output.shape
    N = mm_output.shape[-1]
    if len(mm_output.shape) == 3:
        M = mm_output.shape[0] * mm_output.shape[1]
        mm_output = mm_output.reshape(-1, mm_output.shape[-1])
    else:
        M = mm_output.shape[0]

    # ------------------------------------------#
    # 获取comm_quant_scale_1
    if comm_quant_scale_1 is None :
        abs_tensor = torch.abs(mm_output)
        comm_quant_scale_1_tmp = torch.div(torch.max(abs_tensor, 0)[0], 127).to(torch.float32)
        # 若输入为全0,除0操作置1
        comm_quant_scale_1_rank = torch.ones_like(comm_quant_scale_1_tmp) if torch.all(
            comm_quant_scale_1_tmp == 0) else comm_quant_scale_1_tmp

        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        except OSError:
            pass

        tmp_name = f"{save_path}/comm_quant_scale_1_rank_tmp_{rank}.pt"
        final_name = f"{save_path}/comm_quant_scale_1_rank_{rank}.pt"
        torch.save(comm_quant_scale_1_rank,tmp_name)
        os.replace(tmp_name, final_name)

        for i in range(world_size):
            comm_quant_scale_1_rank_name = f"{save_path}/comm_quant_scale_1_rank_{i}.pt"
            while not os.path.exists(comm_quant_scale_1_rank_name):
                time.sleep(0.5)
            comm_quant_scale_1_rank = torch.load(comm_quant_scale_1_rank_name)
            comm_quant_list.append(comm_quant_scale_1_rank)

        comm_quant_scale_1 = torch.zeros([world_size, N], dtype=torch.float32)

        if len(comm_quant_list) == world_size:
            for i in range(0, world_size):
                comm_quant_scale_1[i, :] = comm_quant_list[i]
            comm_quant_scale_1 = comm_quant_scale_1.max(axis=0)[0]

        save_file(save_path, comm_quant_scale_1, 'bf16', "comm_quant_scale_1", "fp32", 0)

    # step1:quant操作 mm_output / comm_quant_scale_1
    output = (mm_output / comm_quant_scale_1)

    # step2：计算需padding的tensor，并concat拼接成可被all_to_all整除的tensor
    diff_tensor_shape = [math.ceil(M / world_size) * world_size - M, N]
    diff_tensor = torch.zeros(diff_tensor_shape)
    if output.device.type != "cpu":
        diff_tensor = diff_tensor.npu()
    new_output = torch.concat([output, diff_tensor], dim=0)

    # step3: 拼接完的tensor reshape成3维：[world_size, -1, N]
    quant_1_output = new_output.reshape(world_size, -1, N)

    # 转int8
    quant_1_int8_output = torch.clamp(torch.round(quant_1_output), -128, 127).to(torch.int8)

    if is_alltoall:
        # step4: 调用all_to_all接口, all_to_all结果转fp32
        all_to_all_out = torch.zeros_like(quant_1_int8_output)
        dist.all_to_all_single(all_to_all_out, quant_1_int8_output)

        # step5: all_to_all结果做reduce sum,shape：[1,-1,N]
        sum_out = torch.sum(all_to_all_out, dim=0, keepdim=True).to(torch.float32)
    else:
        sum_out = torch.sum(quant_1_int8_output, dim=0, keepdim=True)
    # step6: sum结果做comm_quant_1反量化
    sum_dequant_1_out = sum_out * comm_quant_scale_1

    # -----------------------------------------#
    # 获取comm_quant_scale_2
    if comm_quant_scale_2 is None:
        abs_mul_out = torch.abs(sum_dequant_1_out.reshape(-1, sum_dequant_1_out.shape[-1]))
        comm_quant_scale_2_tmp = torch.div(torch.max(abs_mul_out, 0)[0], 127).to(torch.float32)
        # 若输入为全0,除0操作置1
        comm_quant_scale_2_rank = torch.ones_like(comm_quant_scale_2_tmp) if torch.all(
            comm_quant_scale_2_tmp == 0) else comm_quant_scale_2_tmp

        tmp_name = f"{save_path}/comm_quant_scale_2_rank_tmp_{rank}.pt"
        final_name = f"{save_path}/comm_quant_scale_2_rank_{rank}.pt"
        torch.save(comm_quant_scale_2_rank,tmp_name)
        os.replace(tmp_name, final_name)

        for i in range(world_size):
            comm_quant_scale_2_rank_name = f"{save_path}/comm_quant_scale_2_rank_{i}.pt"
            while not os.path.exists(comm_quant_scale_2_rank_name):
                time.sleep(0.5)
            comm_quant_scale_2_rank = torch.load(comm_quant_scale_2_rank_name)
            comm_quant_list.append(comm_quant_scale_2_rank)

        comm_quant_scale_2 = torch.zeros([world_size, N], dtype=torch.float32)

        if len(comm_quant_list) == 2 * world_size:
            for i in range(world_size, world_size * 2):
                comm_quant_scale_2[i - world_size, :] = comm_quant_list[i]
            comm_quant_scale_2 = comm_quant_scale_2.max(axis=0)[0]

        save_file(save_path, comm_quant_scale_2, 'bf16', "comm_quant_scale_2", "fp32", 0)

    # step7: sum后对1反量化的结果,做comm_quant_2量化，并转int8送进allgather
    quant_2_out = sum_dequant_1_out / comm_quant_scale_2
    quant_2_int8_out = torch.clamp(torch.round(quant_2_out), -128, 127).to(torch.int8)

    all_gather_list = [torch.zeros_like(quant_2_int8_out) for _ in range(world_size)]
    dist.all_gather(all_gather_list, quant_2_int8_out)
    all_gather_out = torch.cat(all_gather_list, dim=0)
    # step 8:all_gather结果做comm_quant_2反量化，并转fp32
    dequant_2_out = (all_gather_out * comm_quant_scale_2).to(torch.float32)
    # step 9:最终结果删除脏数据，并转回3维
    output = dequant_2_out.reshape(-1, dequant_2_out.shape[-1])[0:M, 0:N]
    final_output = output.reshape(original_shape)

    return final_output

def hcom_quant(mm_output, comm_quant_scale_1, comm_quant_scale_2, diff_tensor, all_to_all_out, all_gather_out,
            world_size, is_alltoall, rank=0):
    # 获取原始shape, 若mm_output是三维则reshape成2维进行后续操作
    original_shape = mm_output.shape
    N = mm_output.shape[-1]
    if len(mm_output.shape) == 3:
        M = mm_output.shape[0] * mm_output.shape[1]
        mm_output = mm_output.reshape(-1, mm_output.shape[-1])
    else:
        M = mm_output.shape[0]

    # step1:quant操作 mm_output / comm_quant_scale_1
    output = (mm_output / comm_quant_scale_1)

    # step2：拼接脏数据使得tensor可被all_to_all整除
    new_output = torch.concat([output, diff_tensor], dim=0)

    # step3: 拼接完的tensor reshape成3维：[world_size, -1, N]
    quant_1_output = new_output.reshape(world_size, -1, N)
    # 转int8
    quant_1_int8_output = torch.clamp(torch.round(quant_1_output), -128, 127).to(torch.int8)

    if is_alltoall:
        # step4: 调用all_to_all接口, all_to_all结果转fp32
        dist.all_to_all_single(all_to_all_out, quant_1_int8_output)

        # step5: all_to_all结果做reduce sum,shape：[1,-1,N]
        sum_out = torch.sum(all_to_all_out, dim=0, keepdim=True).to(torch.float32)
    else:
        sum_out = torch.sum(quant_1_int8_output, dim=0, keepdim=True)
    # step6: sum结果做comm_quant_1反量化
    sum_dequant_1_out = sum_out * comm_quant_scale_1
    # step7: sum后对1反量化的结果,做comm_quant_2量化，并转int8送进allgather
    quant_2_out = sum_dequant_1_out / comm_quant_scale_2
    quant_2_int8_out = torch.clamp(torch.round(quant_2_out), -128, 127).to(torch.int8)
    dist._all_gather_base(all_gather_out, quant_2_int8_out)
    # step 8:all_gather结果做comm_quant_2反量化
    dequant_2_out = (all_gather_out * comm_quant_scale_2)
    # step 9:最终结果删除脏数据，并转回3维
    output = dequant_2_out.reshape(-1, dequant_2_out.shape[-1])[0:M, 0:N]
    final_output = output.reshape(original_shape)

    return final_output

def save_file(folder_path, tensor, dtype, tensor_name, bin_name, rank):
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    except OSError:
        pass
    if dtype == 'bf16':
        np.array(tensor.float().cpu()).tofile('{}/{}_{}_{}.bin'.format(folder_path, bin_name, tensor_name, rank))
    else:
        np.array(tensor.cpu()).tofile('{}/{}_{}_{}.bin'.format(folder_path, bin_name, tensor_name, rank))