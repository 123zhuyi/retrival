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
"""
AscendOpTest expect_func（CPU 基准）—— Polar 算子
被 polar_cases.json 的 "expect_func":"<path>/polar_golden.py:polar" 引用。

约束（AscendOpTest README）：
- 参数名/顺序须与算子描述文件(Polar.json)、case 文件 input_desc 一致：input, angle
- 返回 list，元素为 numpy.ndarray；输出 dtype 须为 complex64
- polar(abs, angle) = abs * (cos(angle) + i*sin(angle))，input 与 angle 按 numpy 广播
"""
import numpy as np


def polar(input, angle):
    a = np.asarray(input).astype(np.float32)
    th = np.asarray(angle).astype(np.float32)
    # numpy 广播（input.dim 与 angle.dim 可不一致，与算子 InferShape 对齐）
    a, th = np.broadcast_arrays(a, th)
    out = (a * (np.cos(th) + 1j * np.sin(th))).astype(np.complex64)
    return [np.ascontiguousarray(out)]
