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
import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

@register("ascend_function_aclnn_stft")
class AclnnStftApi(AclnnBaseApi):
    def init_by_input_data(self, input_data):
        """参数处理"""
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.insert(2, input_args[-1])
        input_args.pop()
        output_packages[:] = [input_args[2]]
        return input_args, output_packages

@register("ascend_function_torch_stft")
class TorchStftApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(TorchStftApi, self).__init__(task_result)
        OpsDataset.seed_everything()
        self.change_flag = None


    def __call__(self, input_data: InputDataset, with_output: bool = False):   
        x = input_data.kwargs["self"]
        window = input_data.kwargs["windowOptional"]
        nfft = input_data.kwargs["nFft"]
        hop_length = input_data.kwargs["hopLength"]
        win_length = input_data.kwargs["winLength"]
        normalized = input_data.kwargs["normalized"]
        onesided = input_data.kwargs["onesided"]
        return_complex = input_data.kwargs["returnComplex"]
        if self.output is None:
            output = torch.stft(x,nfft,hop_length=hop_length,win_length=win_length,window=window,center=False,normalized=normalized,onesided=onesided,return_complex=return_complex)
        else:
            if isinstance(self.output, int):
                output = input_data.args[self.output]
            elif isinstance(self.output, str):
                output = input_data.kwargs[self.output]
            else:
                raise ValueError(
                    f"self.output {self.output} value is " f"error"
                )
        return output 