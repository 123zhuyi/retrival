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

@register("ascend_method_torch_unfold_grad")
class MethodTorchUnfoldGrad(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchUnfoldGrad, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"

        input_gradOut = input_data.kwargs["gradOut"].to(device)

        input_inputSizes = np.array(input_data.kwargs["inputSizes"], dtype=np.int64)
        input_inputSizes = tuple(input_inputSizes.tolist())

        dim = input_data.kwargs["dim"]
        size = input_data.kwargs["size"]
        step = input_data.kwargs["step"]
        
        golden = torch.ops.aten.unfold_backward(input_gradOut, input_inputSizes, dim, size, step)

        return golden