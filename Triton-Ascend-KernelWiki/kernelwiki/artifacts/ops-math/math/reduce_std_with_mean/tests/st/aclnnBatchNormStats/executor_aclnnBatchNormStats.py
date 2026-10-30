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


@register("ascend_method_torch_tensor_batch_norm_stats")
class MethodTorchTensorAddcdivApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):

        input = input_data.kwargs["input"]
        length = len(input.shape)
        dimArray = [x for x in range(length) if x != 1]
        mean1 = torch.mean(input, dim=dimArray, keepdim=False)
        mean2 = torch.mean(input, dim=dimArray, keepdim=True)
        variance = torch.mean((input - mean2) ** 2, dim=dimArray, keepdim=False)
        inv_std = 1.0 / torch.sqrt(variance + 1e-5)

        return mean1, inv_std