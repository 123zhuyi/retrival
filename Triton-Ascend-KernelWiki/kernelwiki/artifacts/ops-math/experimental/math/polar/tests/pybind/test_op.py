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
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
import numpy as np
import sys
import warnings

torch.npu.config.allow_internal_format = False

# ============================================================
# Polar: out = input * (cos(angle) + i*sin(angle))
#   input(abs) / angle: float32，支持广播（input.dim 与 angle.dim 可不一致）
#   out: complex64，shape = numpy 广播(input, angle)
#   对齐基准：cann/ops-math math/complex/op_host/op_api/aclnn_polar.cpp
#
# case 设计覆盖：
#   case1 同 shape 基础
#   case2 广播：input 低维 → angle 高维（新增功能主战场）
#   case3 广播：标量 input（numel=1）
#   case4 广播：双向广播 [8,1] × [1,7] → [8,7]
#   case5 大 shape（性能）
#   case6 高维 + inner 非 32B 对齐（[..,269] fp32，每行 1076B）
#   case7 1 元素边界
#   case8 大角度（Sin/Cos 范围归约压力）
# ============================================================
case_data = {
    'case1': {
        'input': np.random.uniform(0, 10,        [2, 6, 10]).astype(np.float32),
        'angle': np.random.uniform(-3.14, 3.14,  [2, 6, 10]).astype(np.float32),
    },
    'case2': {
        'input': np.random.uniform(0, 10,        [4, 1, 8]).astype(np.float32),
        'angle': np.random.uniform(-3.14, 3.14,  [4, 5, 8]).astype(np.float32),
    },
    'case3': {
        'input': np.random.uniform(0, 10,        [1]).astype(np.float32),
        'angle': np.random.uniform(-6.28, 6.28,  [3, 4, 5]).astype(np.float32),
    },
    'case4': {
        'input': np.random.uniform(-10, 10,      [8, 1]).astype(np.float32),
        'angle': np.random.uniform(-3.14, 3.14,  [1, 7]).astype(np.float32),
    },
    'case5': {
        'input': np.random.uniform(0, 100,       [4096, 4096]).astype(np.float32),
        'angle': np.random.uniform(-3.14, 3.14,  [4096, 4096]).astype(np.float32),
    },
    'case6': {
        'input': np.random.uniform(0, 10,        [3, 5, 17, 269]).astype(np.float32),
        'angle': np.random.uniform(-3.14, 3.14,  [3, 5, 17, 269]).astype(np.float32),
    },
    'case7': {
        'input': np.array([2.5], dtype=np.float32),
        'angle': np.array([0.7853981633974483], dtype=np.float32),  # pi/4
    },
    'case8': {
        'input': np.random.uniform(0, 10,        [64, 1024]).astype(np.float32),
        'angle': np.random.uniform(-10000, 10000, [64, 1024]).astype(np.float32),
    },
    # ===== C 泛化数据：各类合法输入场景 =====
    'case9': {   # 8D 满秩（OpDef MAX_DIM=8）同 shape
        'input': np.random.uniform(0, 5,         [2,2,2,2,2,2,2,2]).astype(np.float32),
        'angle': np.random.uniform(-3.14, 3.14,  [2,2,2,2,2,2,2,2]).astype(np.float32),
    },
    'case10': {  # 5D 中间轴广播 input[3,1,5,1,7] × angle[3,4,5,6,7]
        'input': np.random.uniform(0, 10,        [3,1,5,1,7]).astype(np.float32),
        'angle': np.random.uniform(-3.14, 3.14,  [3,4,5,6,7]).astype(np.float32),
    },
    'case11': {  # 双向多轴广播 input[8,1,1] × angle[1,4,5] → [8,4,5]
        'input': np.random.uniform(-10, 10,      [8,1,1]).astype(np.float32),
        'angle': np.random.uniform(-6.28, 6.28,  [1,4,5]).astype(np.float32),
    },
    'case12': {  # 标量 input[1] × 6D angle
        'input': np.array([3.7], dtype=np.float32),
        'angle': np.random.uniform(-3.14, 3.14,  [2,3,2,3,2,3]).astype(np.float32),
    },
    'case13': {  # 负 abs（torch.polar 合法：负 abs 翻号）同 shape
        'input': np.random.uniform(-50, 50,      [128, 257]).astype(np.float32),
        'angle': np.random.uniform(-3.14, 3.14,  [128, 257]).astype(np.float32),
    },
    'case14': {  # 大负角度 + rank 不一致广播 input[5,1] × angle[5,300]
        'input': np.random.uniform(0, 20,        [5,1]).astype(np.float32),
        'angle': np.random.uniform(-50000, -1,   [5,300]).astype(np.float32),
    },
    'case15': {  # 非 32B 对齐 inner + 广播 input[7,1,13] × angle[7,11,13]
        'input': np.random.uniform(0, 8,         [7,1,13]).astype(np.float32),
        'angle': np.random.uniform(-3.14, 3.14,  [7,11,13]).astype(np.float32),
    },
    'case16': {  # 一侧 rank-1 大向量 × 标量 input[20000] × angle[1]
        'input': np.random.uniform(0, 100,       [20000]).astype(np.float32),
        'angle': np.array([1.2345], dtype=np.float32),
    },
}


def verify_result(real_result, golden):
    # 与 AscendOpTest compare_complex 逐行等价（HIT1920/AscendOpTest
    # compare/compare/compare.py）。complex64 在 accuracy_config 无内置默认，
    # 采用 fp32 分量默认 err_threshold = [1e-4, 1e-4]（[绝对偏差, 错误率]）。
    # 判据：实部、虚部各自纯绝对误差 ≤ TOL（无相对回退）；二者同时满足才算正确；
    #       错误数 > size × RATE 判失败。
    TOL = 1e-4
    RATE = 1e-4

    o = real_result.to(torch.complex64).cpu().numpy().reshape(-1)
    g = golden.to(torch.complex64).cpu().numpy().reshape(-1)

    real_diff = np.abs(o.real - g.real)
    imag_diff = np.abs(o.imag - g.imag)
    valid = (real_diff <= TOL) & (imag_diff <= TOL)

    err_num = int(np.sum(~valid))
    total_num = int(o.size)

    if err_num > 0:
        pos = np.argwhere(~valid).reshape(-1)
        print(f"\n[DEBUG] {err_num}/{total_num} elems exceed |Δreal|或|Δimag| > {TOL}")
        print(f"{'idx':<10} | {'NPU':<26} | {'golden':<26} | dRe / dIm")
        print("-" * 90)
        for i in pos[:10]:
            print(f"{int(i):<10} | {str(o[i]):<26} | {str(g[i]):<26} | "
                  f"{real_diff[i]:.3e} / {imag_diff[i]:.3e}")
        print("-" * 90)

    if err_num > total_num * RATE:
        print(f"[ERROR] AscendOpTest-equiv verify FAILED: err {err_num} > {total_num}×{RATE}")
        return False

    print("Test Pass!")
    return True


def golden_polar(np_input, np_angle):
    t_in = torch.from_numpy(np_input).to(torch.float32)
    t_ang = torch.from_numpy(np_angle).to(torch.float32)
    # 显式广播后再调 torch.polar，确保 golden 与广播语义一致
    a, b = torch.broadcast_tensors(t_in, t_ang)
    return torch.polar(a.contiguous(), b.contiguous())   # complex64


class TestCustomOP(TestCase):
    def test_custom_op_case(self, num_str):
        print(f"Running Case: {num_str}")
        caseName = 'case' + num_str
        if caseName not in case_data:
            print(f"Case {caseName} not found!")
            return

        conf = case_data[caseName]
        np_input = conf["input"]
        np_angle = conf["angle"]

        output_cpu = golden_polar(np_input, np_angle)

        input_npu = torch.from_numpy(np_input).npu()
        angle_npu = torch.from_numpy(np_angle).npu()

        try:
            output_npu = custom_ops_lib.custom_op(input_npu, angle_npu)
            if output_npu is None:
                print(f"{caseName} execution failed (Returned None)!")
            else:
                if verify_result(output_npu.cpu(), output_cpu):
                    print(f"{caseName} verify result pass!")
                else:
                    print(f"{caseName} verify result failed!")
        except Exception as e:
            print(f"[CRITICAL] Kernel execution crashed: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_op.py [case_num]")
    else:
        TestCustomOP().test_custom_op_case(sys.argv[1])
