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
import os
import random
import numpy as np
import torch
import ctypes
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

@register("function_npu_add_lora")
class AddLoraApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AddLoraApi, self).__init__(task_result)
        OpsDataset.seed_everything()

    def init_by_input_data(self, input_data: InputDataset):
        seed = self.task_result.case_config.id
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        x_indices = input_data.kwargs['indices']
        x_b1 = input_data.kwargs['weight_a'].shape[0]
        x_indices[x_indices < -1] = 0
        x_indices[x_indices > x_b1] = 0
    
    def bmm_with_mm(self, A, B):
        """
        用 torch.mm 实现 torch.bmm 的功能
        """
        batch_size = A.size(0)
        outputs = []
        for i in range(batch_size):
            # 对每个样本进行矩阵乘法
            out = np.dot(A[i].numpy(), B[i].numpy())#torch.mm(A[i], B[i])  # [m, n] @ [n, p] -> [m, p]
            outputs.append(torch.tensor(out))
        return torch.stack(outputs, dim=0)  # [batch, m, p]

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None
     
        if not with_output:
            if self.device == 'cpu':
                y = input_data.kwargs['input']
                wa_t_all = input_data.kwargs['weight_a']
                wb_t_all = input_data.kwargs['weight_b']
                x = input_data.kwargs['x']
                indices_data = input_data.kwargs['indices']
                layer_idx = input_data.kwargs['layer_idx']
                scale = input_data.kwargs['scale']
                y_offset = input_data.kwargs['y_offset']
                y_slice_size = input_data.kwargs['y_slice_size']
                zeros_wa_4d = torch.zeros((1, wa_t_all.shape[1], wa_t_all.shape[2], wa_t_all.shape[3]), dtype=wb_t_all.dtype)
                zeros_wb_4d = torch.zeros((1, wb_t_all.shape[1], wb_t_all.shape[2], wb_t_all.shape[3]), dtype=wb_t_all.dtype)

                WA = torch.cat((wa_t_all, zeros_wa_4d), dim=0)[indices_data, layer_idx, :, :].transpose(-1, -2)
                WB = torch.cat((wb_t_all, zeros_wb_4d), dim=0)[indices_data, layer_idx, :, :].transpose(-1, -2)
                Z1 = self.bmm_with_mm(x.unsqueeze(1), WA)
                Z2 = self.bmm_with_m(Z1, WB).squeeze() * scale
                y[:, y_offset:y_offset + y_slice_size] += Z2.to(torch.float16)
            return output
        if self.output is None:
            if self.device == 'cpu':
                wa_t_all = input_data.kwargs['weight_a']
                wb_t_all = input_data.kwargs['weight_b']
                y = input_data.kwargs['input']

                x = input_data.kwargs['x']
                indices_data = input_data.kwargs['indices']
                layer_idx = input_data.kwargs['layer_idx']
                scale = input_data.kwargs['scale']
                y_offset = input_data.kwargs['y_offset']
                y_slice_size = input_data.kwargs['y_slice_size']
                zeros_wa_4d = torch.zeros((1, wa_t_all.shape[1], wa_t_all.shape[2], wa_t_all.shape[3]), dtype=wb_t_all.dtype)#.npu()
                zeros_wb_4d = torch.zeros((1, wb_t_all.shape[1], wb_t_all.shape[2], wb_t_all.shape[3]), dtype=wb_t_all.dtype)#.npu()

                WA = torch.cat((wa_t_all, zeros_wa_4d), dim=0)[indices_data, layer_idx, :, :].transpose(-1, -2)
                WB = torch.cat((wb_t_all, zeros_wb_4d), dim=0)[indices_data, layer_idx, :, :].transpose(-1, -2)

                Z1 = self.bmm_with_mm(x.unsqueeze(1), WA)
                Z2 = self.bmm_with_mm(Z1, WB).squeeze() * scale
                y[:, y_offset:y_offset + y_slice_size] += Z2
                output = y

        return output

@register("aclnn_add_lora")
class AclnnAddLoraApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(AclnnAddLoraApi, self).__init__(task_result, backend)

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args[-5] = ctypes.c_int64(input_data.kwargs["layer_idx"])
        input_args[-4] = ctypes.c_double(input_data.kwargs["scale"])
        input_args[-3] = ctypes.c_int64(input_data.kwargs["y_offset"])
        input_args[-2] = ctypes.c_int64(input_data.kwargs["y_slice_size"])
        output_packages[:] = [input_args[0]]
        return input_args, output_packages
    
    def __call__(self):
        super().__call__()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnAddLoraGetWorkspaceSize(const aclTensor *y,const aclTensor *x,const aclTensor *weightB, \
        const aclTensor *indices, const aclTensor *weightAOptional, int64_t layerIdx, double scale, int64_t yOffset, int64_t ySliceSize, const aclTensor *out, \
        uint64_t *workspaceSize, aclOpExecutor **executor)"

@register("aclnn_add_lora_bgmv")
class AclnnAddLoraBgmvApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(AclnnAddLoraBgmvApi, self).__init__(task_result, backend)

    def init_by_input_data(self, input_data: InputDataset):
        from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
        input_args, output_packages = super().init_by_input_data(input_data)
        # weight_a is a nullptr
        input_args.insert(4, TensorPtr())
        input_args[-5] = ctypes.c_int64(input_data.kwargs["layer_idx"])
        input_args[-4] = ctypes.c_double(input_data.kwargs["scale"])
        input_args[-3] = ctypes.c_int64(input_data.kwargs["y_offset"])
        input_args[-2] = ctypes.c_int64(input_data.kwargs["y_slice_size"])
        output_packages[:] = [input_args[0]]
        return input_args, output_packages

    def __call__(self):
        super().__call__()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnAddLoraGetWorkspaceSize(const aclTensor *y,const aclTensor *x,const aclTensor *weightB, \
        const aclTensor *indices, const aclTensor *weightAOptional, int64_t layerIdx, double scale, int64_t yOffset, int64_t ySliceSize, const aclTensor *out, \
        uint64_t *workspaceSize, aclOpExecutor **executor)"

@register("function_npu_add_lora_bgmv")
class AddLoraBgmvApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AddLoraBgmvApi, self).__init__(task_result)
        OpsDataset.seed_everything()

    def init_by_input_data(self, input_data: InputDataset):
        seed = self.task_result.case_config.id
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def bmm_with_mm(self, A, B):
        """
        用 torch.mm 实现 torch.bmm 的功能
        """
        batch_size = A.size(0)
        outputs = []
        for i in range(batch_size):
            # 对每个样本进行矩阵乘法
            out = np.dot(A[i].numpy(), B[i].numpy())#torch.mm(A[i], B[i])  # [m, n] @ [n, p] -> [m, p]
            outputs.append(torch.tensor(out))
        return torch.stack(outputs, dim=0)  # [batch, m, p]

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None

        if not with_output:
            if self.device == 'cpu':#and 'ycov' in self.name:
                y = input_data.kwargs['input']
                wb_t_all = input_data.kwargs['weight_b']
                x = input_data.kwargs['x']
                indices = input_data.kwargs['indices']
                layer_idx = input_data.kwargs['layer_idx']
                scale = input_data.kwargs['scale']
                y_offset = input_data.kwargs['y_offset']
                y_slice_size = input_data.kwargs['y_slice_size']
                zero_w_4d = torch.zeros(1, wb_t_all.shape[1], wb_t_all.shape[2], wb_t_all.shape[3], dtype=wb_t_all.dtype,
                                        device='cpu')
                W = torch.cat((wb_t_all, zero_w_4d), dim = 0)[indices, layer_idx,:,:].transpose(-1, -2)
                Z = self.bmm_with_mm(x.unsqueeze(1), W).squeeze() * scale
                y[:, y_offset:y_offset + y_slice_size] += Z

            return output

        if self.output is None:
            if self.device == 'cpu':
                y = input_data.kwargs['input']
                wb_t_all = input_data.kwargs['weight_b']
                x = input_data.kwargs['x']
                indices = input_data.kwargs['indices']
                layer_idx = input_data.kwargs['layer_idx']
                scale = input_data.kwargs['scale']
                y_offset = input_data.kwargs['y_offset']
                y_slice_size = input_data.kwargs['y_slice_size']
                zero_w_4d = torch.zeros(1, wb_t_all.shape[1], wb_t_all.shape[2], wb_t_all.shape[3], dtype=wb_t_all.dtype,
                                        device=wb_t_all.device)
                W = torch.cat((wb_t_all, zero_w_4d), dim = 0)[indices, layer_idx,:,:].transpose(-1, -2)
                Z = self.bmm_with_mm(x.unsqueeze(1), W).squeeze() * scale
                y[:, y_offset:y_offset + y_slice_size] += Z
                output = y
       
        return output
