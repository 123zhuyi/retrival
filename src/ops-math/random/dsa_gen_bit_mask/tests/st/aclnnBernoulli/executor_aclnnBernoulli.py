#!/usr/bin/env python3
# -- coding: utf-8 --
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
import torch_npu
import numpy as np

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_aclnn_bernoulli")
class MethodAclnnBernoulliApi(BaseApi):

    def __call__(self, task_result: TaskResult):
        super(MethodAclnnBernoulliApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None
        
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 获取yaml中所需参数
        self.tensor = input_data.kwargs['self']
        self.dtype = self.tensor.dtype
        self.prob_ = input_data.kwargs['prob']
        count = 1
        for num in self.tensor.shape:
            count *= num
        self.input = torch.tensor([self.prob_]*count, dtype=self.dtype).reshape(self.tensor.shape)
        
        if self.device == "cpu":
            if self.dtype in [torch.float16, torch.bfloat16]:
                self.input = self.input.to(torch.float32)
        output = torch.bernoulli(self.input)
        if self.device == 'npu' and self.dtype in [torch.float16, torch.bfloat16]:
            output = output.to(torch.float32)
            
        return output

@register("aclnn_bernoulli")
class BernoulliAclnnApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()
    
    def init_by_input_data(self, input_data: InputDataset):
        input_args = []
        output_packages = []
        
        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        
        for name, kwarg in input_data.kwargs.items():
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)
        
        dtype = input_data.kwargs['self'].dtype
        for index, output_data in enumerate(self.task_result.output_info_list):
            output_data.dtype = str(dtype)
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)
        input_args.extend(output_packages)
        
        return input_args, output_packages