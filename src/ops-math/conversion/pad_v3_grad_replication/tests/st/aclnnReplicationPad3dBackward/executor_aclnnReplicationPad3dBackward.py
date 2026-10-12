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

@register("ascend_function_replication_pad3d_backward")
class Pad3dBackward(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        gradOutput = input_data.kwargs["gradOutput"]
        input_self = input_data.kwargs["self"]
        padding = input_data.kwargs["padding"]
        fp16 = 0
        if (gradOutput.dtype == torch.float16):
            fp16 = 1
            gradOutput = gradOutput.to(torch.float32)
            input_self = input_self.to(torch.float32)
        gradinput = torch.ops.aten.replication_pad3d_backward(gradOutput, input_self, padding)
        if (fp16 == 1):
            gradinput = gradinput.to(torch.float16)
        return gradinput