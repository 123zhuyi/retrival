#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
AddMatMatElementsPlus 性能标杆（torch_npu，含数据搬运，与 geir 同条件）
公式: c_out = c * beta + alpha * a * b
"""

import torch
import time


def measure(dtype_name, dtype, n=1000000, iterations=100):
    alpha_val = 1.5
    beta_val = 0.5

    # 预生成 CPU 数据（与 geir 一样从 host 传数据）
    c_cpu = torch.randn(n, dtype=dtype)
    a_cpu = torch.randn(n, dtype=dtype)
    b_cpu = torch.randn(n, dtype=dtype)

    # eager（tensor 已在设备上，不搬数据，作为参考下界）
    c = c_cpu.npu()
    a = a_cpu.npu()
    b = b_cpu.npu()
    alpha = torch.tensor(alpha_val, dtype=dtype).npu()
    beta = torch.tensor(beta_val, dtype=dtype).npu()
    for _ in range(10):
        _ = c * beta + alpha * a * b
    torch.npu.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = c * beta + alpha * a * b
        torch.npu.synchronize()
    eager_ms = (time.time() - start) / iterations * 1000

    # 含数据搬运（与 geir 同条件：每次从 host 搬数据到 device）
    for _ in range(10):
        c2 = c_cpu.npu()
        a2 = a_cpu.npu()
        b2 = b_cpu.npu()
        _ = c2 * beta + alpha * a2 * b2
        torch.npu.synchronize()
    start = time.time()
    for _ in range(iterations):
        c2 = c_cpu.npu()
        a2 = a_cpu.npu()
        b2 = b_cpu.npu()
        _ = c2 * beta + alpha * a2 * b2
        torch.npu.synchronize()
    transfer_ms = (time.time() - start) / iterations * 1000

    print(f"| {dtype_name:6s} | {eager_ms:10.2f} | {transfer_ms:10.2f} |")
    return transfer_ms


if __name__ == "__main__":
    print("=" * 55)
    print(" torch_npu 标杆 (Ascend 910B3, 1M elements)")
    print(" eager = 数据已在设备 | transfer = 含host→device搬运")
    print("=" * 55)
    print("| dtype  | eager(ms) | transfer(ms) |")
    print("|--------|-----------|--------------|")

    results = {}
    for name, dt in [
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ]:
        results[name] = measure(name, dt)

    print()
    print("含数据搬运标杆（与自定义算子 geir 同条件对比）:")
    for name, ms in results.items():
        print(f"  {name}: {ms:.2f} ms/op")
