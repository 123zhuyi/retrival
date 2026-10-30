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
import os
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
import numpy as np

@register("ascend_method_torch_circularpad3dbackward")
class MethodTorchCircularPadBackward3dApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchCircularPadBackward3dApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
            if self.device_id == 0:
                torch.npu.set_compile_mode(jit_compile=False)
            elif self.device_id == 1:
                torch.npu.set_compile_mode(jit_compile=True)
        else:
            device = "cpu"
            
        # #开启确定性计算
        torch.use_deterministic_algorithms(True)

        gradOutput = input_data.kwargs["gradOutput"].to(device)
        input_self = input_data.kwargs["self"].to(device)
        input_padding = np.array(input_data.kwargs["padding"], dtype=np.int64)
        input_padding = tuple(input_padding.tolist())
        self_shape = input_self.shape
        self_dtype = input_self.dtype
        gradIntput = torch.zeros(self_shape, dtype=self_dtype)

        gradOutput.requires_grad = True
        gradIntput.requires_grad = True

        out = torch.nn.functional.pad(gradIntput, input_padding, "circular")
        loss = (gradOutput * out).sum()
        loss.backward()
        golden = gradIntput.grad

        gradOutput.requires_grad = False
        gradIntput.requires_grad = False

        return golden
