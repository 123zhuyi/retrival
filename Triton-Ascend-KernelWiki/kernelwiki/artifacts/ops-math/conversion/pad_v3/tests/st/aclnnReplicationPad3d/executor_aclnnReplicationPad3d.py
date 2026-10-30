#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult


@register("ascend_function_replication_pad3d")
class ReplicationPad3dApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(ReplicationPad3dApi, self).__init__(task_result)
        self.input = None
        self.padding = None
        self.dtype = None

    def init_by_input_data(self, input_data: InputDataset):
        self.input = input_data.kwargs['input']
        self.dtype = self.input.dtype
        self.padding = input_data.kwargs['padding']

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            if self.dtype in [torch.int8, torch.int16, torch.int32, torch.float16]:
                self.input = self.input.to(torch.float32)
            elif self.dtype == torch.int64:
                self.input = self.input.to(torch.float64)
        data_padding = torch.nn.ReplicationPad3d(self.padding)
        output = data_padding(self.input)
        if self.device == "cpu":
            if self.dtype in [torch.int8, torch.int16, torch.int32, torch.float16, torch.int64]:
                output = output.to(self.dtype)
        return output


