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
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

@register("function_aclnn_scale")
class AclnnScale(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x_tensor = input_data.kwargs["x"]
        scale_tensor = input_data.kwargs["scale"]
        bias_tensor = input_data.kwargs["bias"]
        axis = input_data.kwargs["axis"]
        num_axes = input_data.kwargs["numAxes"]
        scale_from_blob = input_data.kwargs["scaleFromBlob"]

        shape_x = list(x_tensor.shape)
        shape_scale = list(scale_tensor.shape)
        shape_bias = list(bias_tensor.shape)
        length_x = len(shape_x)
        if axis < 0:
            axis_ = length_x + axis
        else:
            axis_ = axis
        length_scale = len(shape_scale)
        if scale_from_blob:
            if num_axes == -1:
                shape_left = [1] * axis_
                shape = shape_left + list(shape_scale)
            elif num_axes == 0:
                shape = [1] * length_x
            else:
                left_length = length_x - num_axes - axis_
                shape_left = [1] * axis_
                shape_right = [1] * left_length
                shape = shape_left + list(shape_scale) + shape_right
        else:
            if length_scale == 1 and shape_scale[0] == 1:
                shape = [1] * length_x
            else:
                left_length = length_x - length_scale - axis_
                shape_left = [1] * axis_
                shape_right = [1] * left_length
                shape = shape_left + list(shape_scale) + shape_right
        scale_expand = scale_tensor.reshape(shape)
        if bias_tensor is not None:
            bias_expand = bias_tensor.reshape(shape)
            tmp = torch.mul(x_tensor.to(torch.float32), scale_expand.to(torch.float32))
            res = torch.add(tmp, bias_expand.to(torch.float32)).to(x_tensor.dtype)
        else:
            res = torch.mul(x_tensor.to(torch.float32), scale_expand.to(torch.float32)).to(x_tensor.dtype)
        return res

@register("aclnn_scale")
class AclnnScale(AclnnBaseApi):

    def init_by_input_data(self, input_data: InputDataset):
        import ctypes
        input_args, output_packages = super().init_by_input_data(input_data)
        if sum(input_args[2].shape) == 0:
            input_args[2] = ctypes.c_void_p(0)
            from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
            input_args[2] = TensorPtr()

        return input_args, output_packages