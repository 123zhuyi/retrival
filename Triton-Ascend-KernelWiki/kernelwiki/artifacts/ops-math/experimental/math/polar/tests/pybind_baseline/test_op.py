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
}


def verify_result(real_result, golden):
    # complex64 → 拆 real/imag 当 fp32 比对（AscendOpTest 对复数同口径）
    real_f32 = torch.view_as_real(real_result.to(torch.complex64)).to(torch.float32)
    golden_f32 = torch.view_as_real(golden.to(torch.complex64)).to(torch.float32)

    # fp32 阈值（AscendOpTest 默认阈值，后续以官方工具实际阈值为准，可在此调整）
    rtol, atol = 1e-4, 1e-4

    minimum = 10e-10
    divisor = torch.where(torch.abs(golden_f32) > torch.abs(real_f32),
                          torch.abs(golden_f32), torch.abs(real_f32))
    divisor = torch.where(divisor == 0, torch.full_like(divisor, minimum), divisor)

    abs_diff = torch.abs(real_f32 - golden_f32)
    rel_diff = abs_diff / divisor

    is_close = (abs_diff <= atol) | (rel_diff <= rtol)
    both_nan = torch.isnan(real_f32) & torch.isnan(golden_f32)
    is_close = is_close | both_nan

    err_num = torch.sum(~is_close).item()
    total_num = real_f32.numel()

    if err_num > 0:
        print(f"\n[DEBUG] Found {err_num} errors out of {total_num} (real|imag flattened).")
        print(f"{'Index':<22} | {'NPU (Real)':<15} | {'CPU (Golden)':<15} | {'Abs Diff':<15}")
        print("-" * 78)
        error_indices = torch.nonzero(~is_close)
        for i in range(min(10, err_num)):
            idx = error_indices[i].tolist()
            vr = real_f32[tuple(idx)].item()
            vg = golden_f32[tuple(idx)].item()
            df = abs_diff[tuple(idx)].item()
            print(f"{str(idx):<22} | {vr:<15.6f} | {vg:<15.6f} | {df:<15.6f}")
        print("-" * 78)

    if total_num * rtol < err_num:
        print(f"[ERROR] Result verification failed! Error count: {err_num}")
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
