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


@register("function_aclnnEqual")
class aclnnEqualExecutor(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x_self = input_data.kwargs["self"]
        x_tensor = input_data.kwargs["other"]
        
        data_type = x_self.dtype
        if data_type == torch.float16:
            x_self_np = x_self.numpy()
            x_tensor_np = x_tensor.numpy()
            output = np.array_equal(x_self_np, x_tensor_np)
        else:
            output = torch.equal(x_self, x_tensor)
        return torch.tensor([output])