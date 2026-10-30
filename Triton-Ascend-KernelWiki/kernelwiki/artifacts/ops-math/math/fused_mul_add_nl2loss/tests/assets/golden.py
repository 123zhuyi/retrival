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
"""Golden plugin for FusedMulAddNL2loss.

    y1 = x1 * x3 + x2          (elementwise, x3 标量广播)
    y2 = sum(x1^2 / 2)         (全量 reduce, 标量)

fp64 计算后 cast 到输入 dtype（y1/y2 dtype 均与 x1 一致，对齐 910b 语义）。
"""

import numpy as np

__golden__ = {
    "kernel": {"fused_mul_add_nl2loss": "fused_mul_add_nl2loss_golden"},
    "e2e": {"aclnnFusedMulAddNL2loss": "fused_mul_add_nl2loss_golden"},
}


def fused_mul_add_nl2loss_golden(x1, x2, x3, *args, **kwargs):
    """Golden for fused_mul_add_nl2loss. Parameters follow op proto (x1, x2, x3)。

    kernel 模式传 3 个输入；aclnn(e2e) 模式传全部 5 个 tensor（含 y1/y2 占位），忽略多余参数。
    """
    del args, kwargs
    out_dtype = np.asarray(x1).dtype
    x1_np = np.asarray(x1, dtype=np.float64)
    x2_np = np.asarray(x2, dtype=np.float64)
    x3_np = np.asarray(x3, dtype=np.float64)

    y1 = (x1_np * x3_np + x2_np).astype(out_dtype)
    y2 = np.asarray(np.sum(x1_np**2) * 0.5, dtype=out_dtype)
    return [y1, y2.reshape(1)]  # y2 标量
