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

import sys
import torch
import tensorflow as tf
import numpy as np
import torch.nn.functional as F
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.common.log import Logger
logging = Logger().get_logger()

@register("ascend_aclnn_dropout_backward")
class MethodDropoutBackwardApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodDropoutBackwardApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None


    def __call__(self, input_data: InputDataset, with_output: bool = False):

        def is_double_equal(f1, f2):
            return abs(f1 - f2) <= sys.float_info.epsilon
        
        def revert_array_bit(arr):

            def revert_bit(n):
                result = 0
                for i in range(8):
                    result <<= 1
                    result |= n & 1
                    n >>= 1
                return result
            
            res = []
            for item in arr.flatten():
                res.append(revert_bit(item))
            return np.array(res, dtype=np.uint8).reshape(arr.shape)
        
        grad_output = input_data.kwargs['gradOutput']
        mask = input_data.kwargs['mask']
        scale = float(input_data.kwargs['scale'])

        prob = 1.0 if is_double_equal(scale, 0.0) else (1.0 - 1.0 / scale)
        if is_double_equal(prob, 0.0):
            return grad_output.clone()
        elif is_double_equal(prob, 1.0):
            return torch.zeros_like(grad_output)
        else:
            ori_dtype = grad_output.dtype
            input_x = grad_output
            input_mask = mask
            shape_x = input_x.shape
            size_x = input_x.numel()

            mask_np = revert_array_bit(input_mask.numpy())
            mask_bool = np.unpackbits(mask_np, axis = -1).astype(np.bool_)
            mask_tensor = torch.from_numpy(mask_bool[:size_x]).reshape(shape_x)
            keep_prob = torch.tensor(1.0 - prob, dtype=torch.float32)
            keep_prob_scalar_input_dtype = keep_prob.to(dtype = ori_dtype)
            scale = 1.0 / keep_prob_scalar_input_dtype.to(dtype=torch.float32)

            x_scaled = input_x.to(dtype=torch.float32) * scale
            output = x_scaled * mask_tensor.to(dtype = torch.float32)
            return output.to(dtype = ori_dtype)
