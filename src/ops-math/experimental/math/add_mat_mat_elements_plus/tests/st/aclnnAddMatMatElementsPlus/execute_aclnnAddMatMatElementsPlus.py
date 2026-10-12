# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("execute_aclnnAddMatMatElementsPlus")
class AddMatMatElementsPlusApi(BaseApi):
    """AddMatMatElementsPlus: cOut = c*beta + alpha*(a*b)，逐元素。
    仅定义 cpu 后端 golden；pyaclnn 后端由框架按 aclnn_name=AddMatMatElementsPlus 自动调用。
    """

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "cpu":
            c = input_data.kwargs["c"]
            a = input_data.kwargs["a"]
            b = input_data.kwargs["b"]
            beta = input_data.kwargs["beta"]
            alpha = input_data.kwargs["alpha"]
            return c * beta + alpha * (a * b)
        return None
