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
from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


# aclnn_linalg_cross     
@register("aclnn_linalg_cross")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x1 = input_data.kwargs['x1']
        x2 = input_data.kwargs['x2']
        dim = input_data.kwargs['dim']
        compute_dtype = x1.dtype
        if compute_dtype == torch.float16 or compute_dtype == torch.bfloat16:
            x1 = x1.to(torch.float)
            x2 = x2.to(torch.float)
        y = torch.cross(x1, x2, dim)
        if compute_dtype == torch.float16 or compute_dtype == torch.bfloat16:
            y = y.to(compute_dtype)
        return y           
        
