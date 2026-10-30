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

import random
import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi



@register("function_aclnnOneHot")
class OneHotApi(BaseApi):


    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_tensor=input_data.kwargs["self"]
        numClasses=input_data.kwargs["numClasses"]
        if not input_tensor.is_floating_point():
            input_tensor =input_tensor.to(torch.int64)
        
        input_tensor=torch.clamp(input_tensor,min=0,max=numClasses-1)

        output = torch.nn.functional.one_hot(input_tensor,num_classes=numClasses)
        return output
