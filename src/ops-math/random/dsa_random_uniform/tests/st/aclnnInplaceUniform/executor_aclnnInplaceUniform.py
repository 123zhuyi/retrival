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
    if params["dtype_input"][0] == torch.bfloat16:
        uniform_data = tf.raw_ops.StatelessRandomUniformV2(shape=matrix_shape, key=seed, counter=offset, alg=1,
                                                           dtype=tf.dtypes.bfloat16)
        end = tf.constant(end)
        end = tf.cast(end, tf.dtypes.bfloat16)
        start = tf.constant(start)
        start = tf.cast(start, tf.dtypes.bfloat16)
    elif params["dtype_input"][0] == torch.float16:
        uniform_data = tf.raw_ops.StatelessRandomUniformV2(shape=matrix_shape, key=seed, counter=offset, alg=1,
                                                           dtype=tf.dtypes.float16)
        end = tf.constant(end)
        end = tf.cast(end, tf.dtypes.float16)
        start = tf.constant(start)
        start = tf.cast(start, tf.dtypes.float16)
    else:
        uniform_data = tf.raw_ops.StatelessRandomUniformV2(shape=matrix_shape, key=seed, counter=offset, alg=1)
        end = tf.constant(end)
        end = tf.cast(end, tf.dtypes.float32)
        start = tf.constant(start)
        start = tf.cast(start, tf.dtypes.float32)
    mul_data = tf.multiply(uniform_data, (end - start))
    add_data = tf.add(mul_data, start)
    
    output_data = tf.cast(add_data, dtype)
    if output_data.shape == []:
        output_data = torch.tensor(output_data.numpy())
    else:
        output_data = torch.from_numpy(output_data.numpy())
    
    output_data = output_data.type(params["dtype_input"][0])
    return output_data
    
@register("ascend_aclnn_inplace_uniform")
class MethodAclnnInplaceUniformApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodAclnnInplaceUniformApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 获取yaml中所需参数
        self.Tensor = input_data.kwargs['selfRef']
        self.Tensor_dtype = input_data.kwargs['selfRef'].dtype
        self.from_ = input_data.kwargs['from']
        self.to_ = input_data.kwargs['to']
        self.seed = input_data.kwargs['seed']
        self.offset = input_data.kwargs['offset']

        params = {"from": self.from_, "to": self.to_, "seed": self.seed, "offset": self.offset, "is_contiguous": True,
                  "dtype_input": [self.Tensor_dtype]}
        
        x = uniform_golden(self.Tensor, params)
        
        return x
        
@register("aclnn_inplace_uniform")
class InplaceUniformAclnnApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()
        
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        output_packages[:] = [input_args[0]]
    
        return input_args, output_packages
    
    def after_call(self, output_packages):
        return super().after_call(output_packages)