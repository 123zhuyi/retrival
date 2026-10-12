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

@register("ascend_aclnn_dropout_gen_mask_v2_tensor")
class MethodAclnnDropoutGenMaskV2TensorApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodAclnnDropoutGenMaskV2TensorApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        self.shape = input_data.kwargs["shape"]
        self.prob = input_data.kwargs["prob"]
        self.seed = input_data.kwargs["seedTensor"]
        self.offset = input_data.kwargs["offsetTensor"]
        self.offset2 = input_data.kwargs["offset"]
        self.probDataType = input_data.kwargs["probDataType"]
        self.tensor = torch.ones(self.shape, dtype = self.probDataType).to("npu")

        torch_npu.npu.set_compile_mode(jit_compile = False)
        torch.npu.config.allow_internal_format = True
        output = torch_npu._npu_dropout_gen_mask(self.tensor,
                                                 self.shape,
                                                 self.prob,
                                                 self.seed[0],
                                                 self.offset[0] + self.offset2,
                                                 parallel = False)
        return output
