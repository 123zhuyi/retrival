#!/usr/bin/env python3
# -- coding: utf-8 --
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

@register("ascend_aclnn_dropout_do_mask")
class FunctionalDropoutDoMaskApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionalDropoutDoMaskApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def revert_bit(self, n):
        result = 0
        for i in range(8):
            result <<= 1
            result |= n & 1
            n >>= 1
        return result

    def revert_array_bit(self, arr):
        res = []
        for item in np.array(arr).flatten():
            res.append(self.revert_bit(item))
        return np.array(res, dtype=np.uint8).reshape(np.array(arr).shape)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_x = input_data.kwargs["self"]
        input_mask = input_data.kwargs["mask"]
        input_prob = float(input_data.kwargs["prob"])
        input_dtype = input_x.dtype
        shape_x = input_x.shape
        size_x = input_x.numel()
        if input_prob == 0.0: 
            return input_x.clone()
        if input_prob == 1.0:
            return torch.zeros_like(input_x)

        mask_np = self.revert_array_bit(input_mask.numpy())
        mask_bool = np.unpackbits(mask_np, axis=-1).astype(np.bool_)
        mask_tensor = torch.from_numpy(mask_bool[:size_x]).reshape(shape_x)

        keep_prob = torch.tensor(1.0 - input_prob, dtype=torch.float32)
        keep_prob_scalar_input_dtype = keep_prob.to(dtype=input_dtype)
        scale = 1.0 / keep_prob_scalar_input_dtype.to(dtype=torch.float32)

        x_scaled = input_x.to(dtype=torch.float32) * scale

        output = x_scaled * mask_tensor.to(dtype=torch.float32)
        return output.to(dtype=input_dtype)