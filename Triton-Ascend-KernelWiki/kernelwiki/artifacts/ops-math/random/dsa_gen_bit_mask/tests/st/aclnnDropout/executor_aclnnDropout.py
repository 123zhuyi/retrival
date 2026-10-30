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
import torch_npu
import numpy as np
import tensorflow as tf

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

def uniform_golden(torch_tensor, params):
    seed = []
    offset = [0]
    start = params["from"]
    end = params["to"]
    seed.append(params["seed"])
    offset.append(params["offset"])
    is_contiguous = params["is_contiguous"] if "is_contiguous" in params else True
    if not is_contiguous:
        print("------------非连续操作-------------")
        torch_tensor = torch.transpose(torch_tensor, 0, 1)
    matrix = torch_tensor
    if params["dtype_input"][0] == torch.bfloat16:
        matrix = matrix.float()
    matrix = tf.constant(matrix.cpu())
    dtype = matrix.dtype
    matrix_shape = list(matrix.shape)
    # print("***********************params[\"dtype_input\"][0]=%s****************************" % params["dtype_input"][0])

    uniform_data = tf.raw_ops.StatelessRandomUniformV2(shape=matrix_shape, key=seed, counter=offset, alg=1)
    
    output_data = tf.cast(uniform_data, dtype)
    if output_data.shape == []:
        output_data = torch.tensor(output_data.numpy())
    else:
        output_data = torch.from_numpy(output_data.numpy())
    
    output_data = output_data.type(params["dtype_input"][0])
    return output_data

@register("ascend_aclnn_dropout")
class MethodAclnnDropoutApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodAclnnDropoutApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        self.input = input_data.kwargs['input']
        self.p = input_data.kwargs['p']
        self.train = input_data.kwargs['train']
        self.seed = input_data.kwargs["seed"]
        self.offset = input_data.kwargs["offset"]
        self.shape = self.input.shape

        self.count = 1
        for item in self.shape:
            self.count *= item
        self.tensor = torch.ones([self.count], dtype = torch.float32)
        shape_x = self.input.shape

        inputx = self.input.cpu()
        params = {"from": 0.0, "to": 1.0, "seed": self.seed, "offset": self.offset, "is_contiguous": True,
                  "dtype_input": [torch.float32]}

        x = uniform_golden(self.tensor, params)
        output1 = x.to(torch.float32) >= torch.tensor([self.p], dtype=torch.float32)
        output1 = torch.tensor(output1, dtype=torch.float32).to(torch.uint8)
        output1[self.count:] = 0

        mask = torch.zeros([int(int((self.count + 127) / 128) * 128 / 8)], dtype=torch.uint8)

        mask_tensor = output1[:self.count].reshape(shape_x)

        if self.input.dtype == torch.bfloat16:
            keep_prob = torch.tensor(1.0 - self.p, dtype=torch.float32)
            keep_prob_scalar_input_dtype = keep_prob.to(dtype=self.input.dtype).to(dtype=torch.float32)
            scale = 1.0 / keep_prob_scalar_input_dtype.to(dtype=torch.float32)

            x_scaled = inputx.to(dtype=torch.float32) * scale

            output_data = x_scaled * mask_tensor.to(dtype=self.input.dtype)
        else:
            keep_prob = torch.tensor(1.0 - self.p, dtype=torch.float32)
            keep_prob_scalar_input_dtype = keep_prob.to(dtype=self.input.dtype)
            scale = 1.0 / keep_prob_scalar_input_dtype.to(dtype=self.input.dtype)

            x_scaled = inputx.to(dtype=self.input.dtype) * scale

            output_data = x_scaled * mask_tensor.to(dtype=self.input.dtype)
        output_data[output_data == 0.0] = 0.0

        return output_data.to(dtype=self.input.dtype).to(torch.float32), mask

@register("aclnn_dropout")
class DropoutAclnnApi(AclnnBaseApi):
    def after_call(self, output_packages):
        output1, output2 = super().after_call(output_packages)
        output2.zero_()

        return output1, output2