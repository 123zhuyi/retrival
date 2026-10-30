#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensorStruct, AclFormat
from atk.tasks.dataset.base_dataset import OpsDataset
import numpy as np
import tensorflow as tf

@register("ascend_method_torch_strided_slice_assign_v2")
class MethodTorchStridedSliceAssignV2Api(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MethodTorchStridedSliceAssignV2Api, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):

        input_varRef = input_data.kwargs['varRef']
        variable = tf.Variable(input_varRef)  # 将普通张量转换为可变张量
        input_value = input_data.kwargs['inputValue']
        input_begin = np.array(input_data.kwargs["begin"], dtype=np.int64)
        input_begin = tuple(input_begin.tolist())
        input_end = np.array(input_data.kwargs["end"], dtype=np.int64)
        input_end = tuple(input_end.tolist())
        input_strides = np.array(input_data.kwargs["strides"], dtype=np.int64)
        input_strides = tuple(input_strides.tolist())

        tf.raw_ops.ResourceStridedSliceAssign(
            ref = variable.handle,  
            begin = input_begin,
            end=input_end,
            strides=input_strides,
            value=input_value,
            begin_mask=0,
            end_mask=0,
            ellipsis_mask=0,
            new_axis_mask=0,
            shrink_axis_mask=0
        )
    
        variable = torch.tensor(variable.numpy())

        return variable

@register("aclnn_strided_slice_assign_v2")
class StridedSliceAssignV2AclnnApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        output_packages[:] = [input_args[0]]
        return input_args, output_packages

    def after_call(self, output_packages: AclTensorStruct):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output
