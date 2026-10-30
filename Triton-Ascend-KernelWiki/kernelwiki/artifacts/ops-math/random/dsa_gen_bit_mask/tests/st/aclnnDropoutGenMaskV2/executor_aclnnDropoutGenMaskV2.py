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
import tensorflow as tf
import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

def revert_bit(n):
    result = 0
    for i in range(8):
        result <<= 1
        result |= n & 1
        n >>= 1
    return result

def revert_array_bit(arr):
    res = []
    for item in np.array(arr).flatten():
        res.append(revert_bit(item))
    return np.array(res, dtype=np.uint8).reshape(np.array(arr).shape)

def bitmask_to_list(input_x, input_mask):
    input_dtype = input_x.dtype
    shape_x = input_x.shape
    size_x = input_x.numel()

    mask_np = revert_array_bit(input_mask.numpy())
    mask_bool = np.unpackbits(mask_np, axis=-1).astype(np.bool_)
    mask_tensor = torch.from_numpy(mask_bool[:size_x]).reshape(shape_x)

    output = mask_tensor.to(dtype=torch.float32)
    return output.to(dtype=input_dtype)

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
    if params["dtype_input"][0] == torch.bfloat16:
        uniform_data = tf.raw_ops.StatelessRandomUniformV2(shape=matrix_shape, key=seed, counter=offset, alg=1,
                                                           dtype=tf.dtypes.bfloat16)
    elif params["dtype_input"][0] == torch.float16:
        uniform_data = tf.raw_ops.StatelessRandomUniformV2(shape=matrix_shape, key=seed, counter=offset, alg=1,
                                                           dtype=tf.dtypes.float16)
    else:
        uniform_data = tf.raw_ops.StatelessRandomUniformV2(shape=matrix_shape, key=seed, counter=offset, alg=1)
    
    output_data = tf.cast(uniform_data, dtype)
    if output_data.shape == []:
        output_data = torch.tensor(output_data.numpy())
    else:
        output_data = torch.from_numpy(output_data.numpy())
    
    output_data = output_data.type(params["dtype_input"][0])
    return output_data

@register("ascend_aclnn_dropout_gen_mask_v2")
class MethodAclnnDropoutGenMaskV2Api(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodAclnnDropoutGenMaskV2Api, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def init_by_input_data(self, input_data: InputDataset):
        input_data.kwargs["prob"] = torch.tensor([input_data.kwargs["prob"]], dtype=input_data.kwargs["probDataType"]).to(torch.float32).numpy()[0]

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        self.shape = input_data.kwargs["shape"]
        self.prob = input_data.kwargs["prob"]
        self.seed = input_data.kwargs["seed"]
        self.offset = input_data.kwargs["offset"]
        self.probDataType = input_data.kwargs["probDataType"]

        self.count = 1
        for item in self.shape:
            self.count *= item
        self.tensor = torch.ones([self.count], dtype = self.probDataType)
        
        params = {"from": 0.0, "to": 1.0, "seed": self.seed, "offset": self.offset, "is_contiguous": True,
                  "dtype_input": [self.probDataType]}

        x = uniform_golden(self.tensor, params)
        output1 = x.to(torch.float32) >= torch.tensor([self.prob], dtype=torch.float32)
        output1 = torch.tensor(output1, dtype=torch.float32).to(torch.uint8)
        output1[self.count:] = 0

        return output1.to(torch.uint8).contiguous()

@register("aclnn_dropout_gen_mask_v2")
class DropoutGenMaskV2AclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_data.kwargs["prob"] = torch.tensor([input_data.kwargs["prob"]], dtype=input_data.kwargs["probDataType"]).to(torch.float32).numpy()[0]
        self.offset = input_data.kwargs["offset"]
        self.seed = input_data.kwargs["seed"]
        self.shape = input_data.kwargs["shape"]
        self.probDataType = input_data.kwargs["probDataType"]
        input_args, output_packages = super().init_by_input_data(input_data)

        self.count = 1
        for item in self.shape:
            self.count *= item
        self.tensor = torch.ones([self.count], dtype = self.probDataType).to("npu")
        self.task_result.output_info_list[0].shape = [int(int((self.count + 127) / 128) * 128 / 8)]
        self.task_result.output_info_list[0].stride = [1]
        output = self.backend.convert_output_data(self.task_result.output_info_list[0], 0)
        output_packages[0] = output[0]
        input_args[-1] = output_packages[0]

        return input_args, output_packages

    def after_call(self, output_packages):
        output1 = super().after_call(output_packages)
        output1[0] = bitmask_to_list(self.tensor, output1[0].cpu()).to(torch.uint8).npu()
        output1[0][self.count:] = 0

        return output1