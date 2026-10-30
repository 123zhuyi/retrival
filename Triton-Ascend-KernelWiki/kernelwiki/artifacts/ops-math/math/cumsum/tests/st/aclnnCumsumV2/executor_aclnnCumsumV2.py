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

import numpy as np
import torch
import tensorflow as tf
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("function_aclnnCumsumV2")
class aclnnCumsumV2Executor(BaseApi):

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """
        :param input_data:
        :param with_output:
        :return:
        """
        x_x = input_data.kwargs.get("x")
        x_axis = input_data.kwargs.get("axis")
        x_exclusive = input_data.kwargs.get("exclusive")
        x_reverse = input_data.kwargs.get("reverse")

        output = torch.from_numpy(tf.math.cumsum(x_x.numpy(), axis=x_axis, exclusive=x_exclusive, reverse=x_reverse).numpy())
        return output
