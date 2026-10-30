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


@register("aclnn_replication_pad1d")
class ReplicationPad1d(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_tensor = input_data.kwargs["input"]
        padding = input_data.kwargs["padding"]
        
        if isinstance(padding, int):
            padding = (padding, padding)
        
        replication_pad1d = torch.nn.ReplicationPad1d(padding)

        if input_tensor.dtype == torch.float16:
            input_tensor = input_tensor.to(torch.float32)

        padded_tensor = replication_pad1d(input_tensor)

        return padded_tensor.to(torch.float16) if input_data.kwargs["input"].dtype == torch.float16 else padded_tensor