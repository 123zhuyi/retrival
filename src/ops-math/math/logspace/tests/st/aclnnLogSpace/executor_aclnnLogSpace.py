# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("function_aclnnLogSpace")
class AclnnLogSpaceExecutor(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 获取所有必需参数
        start = input_data.kwargs["start"]
        end = input_data.kwargs["end"]
        steps = input_data.kwargs["steps"]
        base = input_data.kwargs["base"]  # JSON中是fp64(double)，需要转换为float
        # 将base从double转换为float
        base_float = float(base)  # 显式转换为Python float
        # 调用PyTorch的logspace，使用转换后的float类型base
        output = torch.logspace(start, end, steps, base=base_float)
        return output
