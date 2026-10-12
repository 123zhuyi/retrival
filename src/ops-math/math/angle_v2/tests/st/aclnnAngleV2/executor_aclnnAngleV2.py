#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""ATK golden executor for AngleV2 (torch.angle).

Operator semantics (truth source = torch.angle):
    y = angle(x)
  - Input: real or complex tensor
  - Output: float tensor with phase angle in range [-pi, pi]

This file registers ONLY the golden (cpu reference). The device-under-test (pyaclnn) side is
resolved by the built-in AclnnBaseApi via the case json's ``aclnn_name: AngleV2``
(-> aclnnAngleV2GetWorkspaceSize / aclnnAngleV2).
"""

import torch

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("aclnn_angle_v2")
class AclnnAngleV2Api(BaseApi):
    """Golden: y = torch.angle(x)"""

    def __call__(self, input_data, with_output: bool = False):
        kw = input_data.kwargs
        if kw and "x" in kw:
            x = kw["x"]
        else:
            args = input_data.args
            x = args[0]

        # torch.angle computes phase angle for complex tensors
        # For real tensors, returns 0 for positive, pi for negative
        return torch.angle(x)
