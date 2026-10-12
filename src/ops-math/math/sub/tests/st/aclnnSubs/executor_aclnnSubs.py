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
from atk.tasks.dataset.base_dataset import OpsDataset
import numpy as np


@register("function_aclnnSubs")
class aclnnSubsExecutor(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x_input = input_data.kwargs["input"]
        x_other = input_data.kwargs["other"]
        x_alpha = input_data.kwargs["alpha"]
        
        data_type = x_input.dtype
        if data_type == torch.bool:
            x_input_np = x_input.numpy()
            x_other_np = x_other.numpy()
            output = torch.from_numpy(np.logical_xor(x_input_np, np.multiply(x_other_np, x_alpha)))
        else:
            output = torch.sub(x_input, x_other, alpha=x_alpha)
        return output