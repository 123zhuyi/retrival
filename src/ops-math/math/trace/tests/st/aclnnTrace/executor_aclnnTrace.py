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


@register("aclnn_trace")
class AclnnTraceApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        output = None
        if self.device != "npu":
            input_dtype = input_data.args[0].dtype
            if input_dtype == torch.bool:
                output = torch.trace(input_data.args[0].to(torch.int64))
            elif input_dtype in [torch.float16, torch.bfloat16]:
                output = torch.trace(input_data.args[0].to(torch.int64)).to(input_dtype)
            else:
                output = torch.trace(input_data.args[0])
        else:
            output = torch.trace(input_data.args[0])
        return output