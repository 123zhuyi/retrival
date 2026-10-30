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

FLOAT16_DIGITS = 11
BF16_DIGITS = 8
FLOAT32_DIGITS = 24
DOUBLE_DIGITS = 53

type2digits = {}
type2digits[torch.float16] = FLOAT16_DIGITS
type2digits[torch.bfloat16] = BF16_DIGITS
type2digits[torch.float32] = FLOAT32_DIGITS
type2digits[torch.float64] = DOUBLE_DIGITS

def update_from(from_value, scalar_type=torch.float32):
    # 判断是否为浮点数数据类型
    if not scalar_type.is_floating_point:
        return from_value
    
    tmp_value = torch.tensor(from_value + 1, dtype=torch.int64)
    from_plus_1 = tmp_value.to(scalar_type).to(torch.int64)
    if from_plus_1 < from_value:
        from_ = abs(from_value + 1)
        n = 0
        from_ >>= 1
        while from_:
            n += 1
            from_ >>= 1
        digits = type2digits[scalar_type] # 获取浮点类型的尾数位数
        adjustment = 1 << (n - digits + 1)
        from_value = int(from_plus_1 + adjustment)
        
    return from_value

def update_to(to_value, scalar_type=torch.float32):
    # 判断是否为浮点数数据类型
    if not scalar_type.is_floating_point:
        return to_value
    
    tmp_value = torch.tensor(to_value - 1, dtype=torch.int64)
    to_minus_1 = tmp_value.to(scalar_type).to(torch.int64)
    if to_minus_1 >= to_value:
        to_ = abs(to_value - 1)
        n = 0
        to_ >>= 1
        while to_:
            n += 1
            to_ >>= 1
        digits = type2digits[scalar_type] # 获取浮点类型的尾数位数
        adjustment = 1 << (n - digits + 1)
        to_value = int(to_minus_1 - adjustment)
        
    return to_value

def random_golden(torch_tensor, params):
    seed = []
    offset = [0]
    start = params["from"]
    end = params["to"]
    seed.append(params["seed"])
    offset.append(params["offset"])
    is_contiguous = params["is_contiguous"] if "is_contiguous" in params else True
    
    start = update_from(start, params["dtype_input"][0])
    end = update_to(end, params["dtype_input"][0])
    if not is_contiguous:
        print("------------非连续操作-------------")
        torch_tensor = torch.transpose(torch_tensor, 0, 1)
    matrix = torch_tensor
    if params["dtype_input"][0] == torch.bfloat16:
        matrix = matrix.float()
    matrix = tf.constant(matrix)
    matrix_shape = list(matrix.shape)
    uniform_data = tf.raw_ops.StatelessRandomUniformV2(shape=matrix_shape, key=seed, counter=offset, alg=1)
    mul_data = tf.multiply(uniform_data, (end - start))
    add_data = tf.add(mul_data, start)
    if params["dtype_input"][0] == torch.bool:
        add_data = tf.math.round(add_data)
    output_data = tf.cast(add_data, tf.int64)
    output_data = output_data.numpy()
    output_data = torch.from_numpy(output_data)
    output_data = output_data.to(dtype=params["dtype_input"][0])
    return output_data
    
@register("ascend_aclnn_inplace_random_tensor")
class MethodAclnnInplaceRandomTensorApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodAclnnInplaceRandomTensorApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 获取yaml中所需参数
        self.Tensor = input_data.kwargs['selfRef']
        self.Tensor_dtype = input_data.kwargs['selfRef'].dtype
        self.from_ = input_data.kwargs['from']
        self.to_ = input_data.kwargs['to']
        self.seed = input_data.kwargs['seedTensor'].cpu().numpy()
        self.offset = input_data.kwargs['offsetTensor'].cpu().numpy()
        self.offset2 = input_data.kwargs['offset']

        params = {"from": self.from_, "to": self.to_, "seed": self.seed[0], "offset": self.offset[0] + self.offset2, "is_contiguous": True,
                  "dtype_input": [self.Tensor_dtype]}
        
        x = random_golden(self.Tensor, params)
        
        return x
        
@register("aclnn_inplace_random_tensor")
class InplaceRandomTensorAclnnApi(AclnnBaseApi):
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

