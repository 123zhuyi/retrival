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
import torch_npu
import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_aclnn_rand_perm")
class MethodAclnnRandpermApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodAclnnRandpermApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def init_by_input_data(self, input_data: InputDataset):
        input_data.kwargs["offset"] = 0

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        self.n = input_data.kwargs["n"]
        self.seed = input_data.kwargs["seed"]

        torch.manual_seed(self.seed)
        output = torch.randperm(self.n, device="npu")

        return output
