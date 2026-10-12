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
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

import os
os.environ["PYTORCH_NO_NPU_MEMORY_CACHING"] = "1"
@register("ascend_function_grouped_bias_add_grad")
class MethodTorchGroupedBiasAddGradApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchGroupedBiasAddGradApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        grad_y = input_data.kwargs["gradY"]
        group_idx = input_data.kwargs["groupIdxOptional"]
        grad_bias = None
        if group_idx == None:
            grad_y_dtype = grad_y.dtype
            if grad_y_dtype != torch.float32:
                grad_y = grad_y.to(torch.float32)

            grad_bias = torch.sum(grad_y, 1)
            return grad_bias.to(grad_y_dtype)
        grad_y_dtype = grad_y.dtype
        if grad_y_dtype != torch.float32:
            grad_y = grad_y.to(torch.float32)
        grad_bias = torch.tensor([]).to(grad_y.device)
        for i, num in enumerate(group_idx):
            if i == 0:
                x = grad_y[:num, :]
                tmp = torch.sum(x, 0, keepdim=True)
                grad_bias = torch.cat((grad_bias, tmp), 0)
            else:
                x = grad_y[group_idx[i-1]:num, :]
                tmp = torch.sum(x, 0, keepdim=True)
                grad_bias = torch.cat((grad_bias, tmp), 0)

        return grad_bias.to(grad_y_dtype)

    
    def init_by_input_data(self, input_data: InputDataset, with_output: bool = False):
        grad_y = input_data.kwargs["gradY"]
        groupIdxOptional = input_data.kwargs["groupIdxOptional"]
        if groupIdxOptional.shape[0] != 0:
            if self.device == 'pyaclnn':
                input_data.kwargs['groupIdxOptional'] = torch.linspace(grad_y.shape[0] // groupIdxOptional.shape[0], grad_y.shape[0], steps=int(groupIdxOptional.size(0))).round().to(groupIdxOptional.dtype).npu()
            else:
                input_data.kwargs['groupIdxOptional'] = torch.linspace(grad_y.shape[0] // groupIdxOptional.shape[0], grad_y.shape[0], steps=int(groupIdxOptional.size(0))).round().to(groupIdxOptional.dtype).cpu()
        else:
            input_data.kwargs["groupIdxOptional"] = None