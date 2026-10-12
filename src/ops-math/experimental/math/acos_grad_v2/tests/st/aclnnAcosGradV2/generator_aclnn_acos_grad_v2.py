#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------


from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


@GENERATOR_REGISTRY.register("ascend_aclnn_acos_grad_v2")
class AcosGradV2Generator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        # z = -dy / sqrt(1 - y^2): y and dy must share the same shape and dtype
        case_config.inputs[1].dtype = case_config.inputs[0].dtype
        case_config.inputs[1].shape = case_config.inputs[0].shape
        return case_config
