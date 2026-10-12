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
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
import numpy as np

@register("ascend_method_torch_replication_pad2d")
class ReplicationPad2dApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(ReplicationPad2dApi, self).__init__(task_result)
        self.input_tensor = None
        self.dtype = None


    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == 'cpu':
            self.input_tensor = input_data.kwargs["self"]
            self.dtype = self.input_tensor.dtype
            input_padding = np.array(input_data.kwargs["padding"], dtype=np.int64)
            input_padding = tuple(input_padding.tolist())

            if self.device == "cpu":
                if self.dtype in [torch.int8, torch.int16, torch.int32, torch.float16]:
                    self.input_tensor = self.input_tensor.to(torch.float32)
                elif self.dtype == torch.int64:
                    self.input_tensor = self.input_tensor.to(torch.float64)
        pad = torch.nn.ReplicationPad2d(input_padding)
        output = pad(self.input_tensor)

        if self.device == "cpu":
            if self.dtype in [torch.int8, torch.int16, torch.int32, torch.float16, torch.int64]:
                output = output.to(self.dtype)
        return output