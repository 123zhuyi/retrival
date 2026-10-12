#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi


def reference(input_data: InputDataset):
    """Golden: z = -dy / sqrt(1 - y^2)"""
    y = input_data.kwargs["y"]
    dy = input_data.kwargs["dy"]
    return -dy / torch.sqrt(1 - y * y)


@register("ascend_acos_grad_v2")
class TorchAcosGradV2(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        return reference(input_data)


@register("aclnn_acos_grad_v2")
class AclnnAcosGradV2(AclnnBaseApi):
    def __call__(self):
        super().__call__()
