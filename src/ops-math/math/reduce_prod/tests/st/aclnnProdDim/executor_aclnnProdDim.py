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
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("executor_aclnnProdDim")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            map = {
                0: torch.float32,
                1: torch.float16,
                2: torch.int8,
                3: torch.int32,
                4: torch.uint8,
                6: torch.int16,
                9: torch.int64,
                11: torch.float64,
                12: torch.bool,
                16: torch.complex64,
                17: torch.complex128,
                27: torch.bfloat16
            }
            output = torch.prod(input_data.kwargs["self"], 
                                dim=input_data.kwargs["dim"], 
                               keepdim=input_data.kwargs["keepDim"], 
                               dtype=map.get(input_data.kwargs["dtype"]))
        return output