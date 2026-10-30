#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

# 定义测试参数组合
TEST_PARAMS = {
    # 基础场景
    "base_default": {
        "in_layout": ["BNSD"],
        "dtype": [torch.float16],
        "head_dim": [128], 
        "kv_seq": [512],
        "q_seq": [128],   
        "kv_head_num": [32],
        "q_head_num": [64],
        "batch_size": [24],
        "rope": [0],
    }, # q(24, 64, 128, 128), k(24, 32, 512, 128), v(24, 32, 512, 128)

    "mla": {
        "in_layout": ["BNSD"],
        "dtype": [torch.float16],
        "head_dim": [128],
        "kv_seq": [512],
        "q_seq": [64],   
        "kv_head_num": [1],
        "q_head_num": [5],
        "batch_size": [4],
        "rope": [64],
    }, # q(4, 5, 64, 128), k(4, 1, 512, 128), v(4, 1, 512, 128), rope=64
}

# 按需选择要启用的测试参数（例如默认启用所有）
ENABLED_PARAMS = [TEST_PARAMS["base_default"]] # 或指定子集，如[TEST_PARAMS["base"], TEST_PARAMS["mixed"]], TEST_PARAMS