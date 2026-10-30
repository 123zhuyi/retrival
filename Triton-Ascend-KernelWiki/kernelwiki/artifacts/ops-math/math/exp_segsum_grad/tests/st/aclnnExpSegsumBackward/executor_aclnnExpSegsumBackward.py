#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from abc import ABC

import copy
import torch
from einops import repeat

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.function_api import FunctionApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.configs.results_config import OutputData

logging = Logger().get_logger()

try:
    import torch_npu
except Exception as e:
    logging.warning("import torch_npu failed!!!")

@register("segsum_backward")
class FunctionSegsumApi(FunctionApi):
    def segsum_backward(self,gradout,output):
        T = gradout.size(-1)
        y = gradout * output
        mask = torch.tril(torch.ones(T, T, device=gradout.device, dtype=bool), diagonal=0)
        y = y.masked_fill(~mask, 0)
        y = torch.flip(y, [-2])
        y = torch.cumsum(y, dim = -2)
        y = torch.flip(y, [-2])
        mask = torch.tril(torch.ones(T, T, device=gradout.device, dtype=bool), diagonal=-1)
        y = y.masked_fill(~mask, 0)
        gradin = torch.sum(y, dim = -1, keepdim=False)
        return gradin

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        grad = input_data.kwargs["gradOut"]
        output = input_data.kwargs["gradSelf"]
        dtype = grad.dtype
        if self.device in ['cpu', 'gpu'] and (grad.dtype == torch.float16 or grad.dtype == torch.bfloat16):
            gradInput = self.segsum_backward(grad.to(torch.float32), output.to(torch.float32)).to(dtype)
        elif self.device == 'npu'and grad.dtype == torch.bfloat16:
            gradInput = self.segsum_backward(grad.to(torch.float32), output.to(torch.float32)).to(dtype)
        else:
            gradInput = self.segsum_backward(grad, output)
        return gradInput
