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
import onnx
import onnxruntime
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from tasks.api_execute.customs.Onnx.utils import trans_tensor_to_numpy, expect, NP_TYPE_TO_TENSOR_TYPE


@register("onnx_global_max_pool")
class OnnxGlobalMaxPool(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(OnnxGlobalMaxPool, self).__init__(task_result)
        self.session = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        """
        :param input_data:
        :param with_output:
        :return:
        """
        input_self = trans_tensor_to_numpy(input_data.kwargs.get("x"))
        output = self.session.run(None, {"input_self": input_self})
        return torch.from_numpy(output[0])

    def init_by_input_data(self, input_data: InputDataset):
        node = onnx.helper.make_node(
            "GlobalMaxPool",
            inputs=["input_self"],
            outputs=["y"],
        )
        input_self_type = input_data.kwargs.get("x").dtype
        model = expect(
            node,
            input_types=[NP_TYPE_TO_TENSOR_TYPE[input_self_type], 7],
            output_types=[NP_TYPE_TO_TENSOR_TYPE[input_self_type]],
            name="node",
        )
        for opset in model.opset_import:
            if opset.domain == "":
                opset.version = 11
        self.session = onnxruntime.InferenceSession(model.SerializeToString())


