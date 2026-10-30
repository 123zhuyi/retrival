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

@register("ascend_method_torch_reflectionpad2d")
class MethodTorchReflectionPad2dApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchReflectionPad2dApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == 'cpu':
            input_tensor = input_data.kwargs["self"]
            input_padding = np.array(input_data.kwargs["padding"], dtype=np.int64)
            input_padding = tuple(input_padding.tolist())
            fp16 = 0
            if input_tensor.dtype == torch.float16:
                fp16 = 1
                input_tensor = input_tensor.to(torch.float32)

            pad = torch.nn.ReflectionPad2d(input_padding)
            output = pad(input_tensor)

            if (fp16 == 1):
                output = output.to(torch.float16)

        return output