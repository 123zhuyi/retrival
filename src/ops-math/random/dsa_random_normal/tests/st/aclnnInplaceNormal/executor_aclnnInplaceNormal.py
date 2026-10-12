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
import tensorflow as tf

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

def normal_golden(torch_tensor, params):
    if tf.__version__ >= '2':
        tf.compat.v1.enable_eager_execution()
    seed = []
    offset = [0]
    mean = params["mean"]
    std = params["std"]
    seed.append(params["seed"])
    offset.append(params["offset"])
    is_contiguous = params["is_contiguous"] if "is_contiguous" in params else True
    if not is_contiguous:
        print("------------非连续操作-------------")
        torch_tensor = torch.transpose(torch_tensor, 0, 1)
    matrix = torch_tensor
    if params["dtype_input"][0] == torch.bfloat16:
        dtype = tf.dtypes.float32
    elif params["dtype_input"][0] == torch.float16:
        dtype = tf.dtypes.float16
    else:
        dtype = tf.dtypes.float32
    matrix_shape = list(matrix.shape)
    normal_data = tf.raw_ops.StatelessRandomNormalV2(shape=matrix_shape, key=seed, counter=offset, alg=1)
    mul_data = tf.multiply(normal_data, std)
    add_data = tf.add(mul_data, mean)
    if tf.__version__ >= '2':
        output_data = tf.cast(add_data, dtype).numpy()
        output_data = torch.from_numpy(output_data)
    else:
        sess = tf.compat.v1.Session()
        output_data_tf = sess.run(output_data)
        output_data = torch.from_numpy(output_data_tf)
    output_data = output_data.to(params["dtype_input"][0])
    return output_data
    
@register("ascend_aclnn_inplace_normal")
class MethodAclnnInplaceNormalApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodAclnnInplaceNormalApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 获取yaml中所需参数
        self.tensor_ = input_data.kwargs['selfRef']
        self.tensor_dtype_ = input_data.kwargs['selfRef'].dtype
        self.mean_ = input_data.kwargs['mean']
        self.std_ = input_data.kwargs['std']
        self.seed_ = input_data.kwargs['seed']
        self.offset_ = input_data.kwargs['offset']

        params = {"mean": self.mean_, "std": self.std_, "seed": self.seed_, "offset": self.offset_, "is_contiguous": True,
                  "dtype_input": [self.tensor_dtype_]}
        
        x = normal_golden(self.tensor_, params)
        
        return x
        
@register("aclnn_inplace_normal")
class InplaceNormalAclnnApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()
        
    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        output_packages[:] = [input_args[0]]
    
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))

        return output

