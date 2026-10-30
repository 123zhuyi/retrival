#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
ST 测试 —— SortWithIndex（experimental/math/sort_with_index，ascend910b）ACLNN 两段式 PyTorch 接入

================================================================================
本 PyTorch ST 的两种运行模式：
  - CPU Golden 自测 + mock 比对闭环（始终运行，无需 NPU）：
    验证 910B NaN-开头 golden 逻辑（golden.py）与 bitwise/isnan/不变量比对链路（compare.py）。
  - NPU 实跑（torch_npu + NPU + 已加载 libtorch_adapter 时）：
    调用 torch.ops.sort_with_index.forward，取回 CPU 与 golden 比对；
    torch_npu 不可用时自动回退 mock 闭环。

数据流（torch_npu 可用时）：
   PyTorch Tensor(NPU) -> libtorch_adapter.so -> ACLNN aclnnSortWithIndex -> NPU(y, sorted_index)
   CPU golden(golden.py)  <- compare.py(bitwise+isnan+不变量) ->  NPU 实际结果（取回 CPU）

用例覆盖（对应 C++ ST 全量 GetFullCases，4 组 dtype + shape/属性 + 边界 + extreme）：
   A. 每 dtype 基础 shape × descending × stable
   B. 多 rank（rank0–8）+ axis(-1, rank-1)
   C. 单 tile 内多轴长（<= 单 tile 上限：fp16~3008/fp32~2816/bf16~2816/int32~2560）
   D. 边界：空 tensor [0]/[3,0]/[0,8]
   E. extreme：NaN(升序落开头)/±Inf/全零/全相等/±0
   F. 确定性：含 ties 的重复执行
   负向：x≠index shape、axis 非最后一维、轴超上限（仅 NPU 可用时构造）

用法：
   python3 test.py                                  # 仅 golden 自测 + mock 比对闭环（无 NPU）
   python3 test.py --lib ./build/libtorch_adapter.so  # NPU 可用时加载并实跑
================================================================================
"""

import argparse
import os
import sys

import torch

from golden import compute_golden_sort_with_index, test_golden_correctness
from compare import compare_results


# ============================================================================
# dtype 映射
# ============================================================================

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
}

# 各 value dtype 单 tile 安全 N 上限（约值；以运行时拒绝为准）。
N_CAP = {"float16": 3008, "float32": 2816, "bfloat16": 2816, "int32": 2560}


# ============================================================================
# 数据构造（对齐 C++ ST BuildRowFloats：严格唯一有限值 + extreme patterns）
# ============================================================================

def _gen_finite_row(dtype: str, n: int) -> torch.Tensor:
    """生成一行严格唯一、该 dtype 可精确表示的有限值（大 N 无 ties，indices 可精确比对）。"""
    if dtype == "float16":
        # fp16 正规正数位型 0x3C00=1.0；+k 单调递增、互异（k<n<=3008 不越 Inf 0x7C00）。
        bits = (0x3C00 + torch.arange(n, dtype=torch.int32)).to(torch.int16)
        return bits.view(torch.float16)
    if dtype == "bfloat16":
        bits = (0x3F80 + torch.arange(n, dtype=torch.int32)).to(torch.int16)
        return bits.view(torch.bfloat16)
    if dtype == "float32":
        v = torch.arange(n, dtype=torch.float32) - (n // 2)
        return v
    # int32：(k - n/2) 整数，|x|<=2^24 恒成立。
    return (torch.arange(n, dtype=torch.int32) - (n // 2)).to(torch.int32)


def _build_row(dtype: str, pattern: str, n: int, row_seed: int) -> torch.Tensor:
    """按 pattern 构造一行 value（float 语义），返回目标 dtype tensor。"""
    nan = float("nan")
    pinf = float("inf")
    ninf = float("-inf")
    tdt = DTYPE_MAP[dtype]

    if pattern == "scalar":
        v = 42.0 if dtype == "int32" else 13.5
        return torch.tensor([v], dtype=tdt) if n == 1 else torch.full((n,), v, dtype=tdt)
    if pattern == "all_zero":
        return torch.zeros(n, dtype=tdt)
    if pattern == "all_same":
        v = 5.0 if dtype == "int32" else 1.0
        return torch.full((n,), v, dtype=tdt)
    if pattern == "signed_zero_mix":
        base = torch.tensor([0.0 if k % 2 == 0 else -0.0 for k in range(n)], dtype=torch.float32)
        return base.to(tdt)
    if pattern == "with_ties":
        base = torch.tensor([float((k // 2) + 1) for k in range(n)], dtype=torch.float32)
        return base.to(tdt)

    # extreme（NaN/Inf）仅浮点 dtype；构造在 float 域再转目标 dtype。
    finite = _gen_finite_row(dtype, n).to(torch.float32)
    if pattern == "inject_nan":
        finite[n // 2] = nan
    elif pattern == "single_pos_inf":
        finite[n // 3] = pinf
    elif pattern == "single_neg_inf":
        finite[n * 2 // 3] = ninf
    elif pattern == "nan_inf_combo" and n >= 4:
        finite[0] = ninf
        finite[n // 3] = pinf
        finite[n // 2] = nan
        finite[n - 1] = -0.0
    elif pattern == "random_unique":
        return _gen_finite_row(dtype, n)  # 直接返回原 dtype（保位型精确）
    return finite.to(tdt)


def build_case_inputs(dtype: str, shape, pattern: str):
    """构造 (x, index)：value 按 pattern，index 每行 0..N-1（int32）。"""
    tdt = DTYPE_MAP[dtype]
    if len(shape) == 0:  # rank0 标量
        x = _build_row(dtype, pattern if pattern != "random_unique" else "scalar", 1, 0)[0]
        return x.to(tdt), torch.tensor(0, dtype=torch.int32)

    n = shape[-1]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    total = rows * n

    if total == 0:  # 空 tensor
        return torch.empty(shape, dtype=tdt), torch.empty(shape, dtype=torch.int32)

    x_rows = []
    idx_rows = []
    for r in range(rows):
        x_rows.append(_build_row(dtype, pattern, n, r + 1).reshape(n))
        idx_rows.append(torch.arange(n, dtype=torch.int32))
    x = torch.stack(x_rows).reshape(shape).to(tdt)
    index = torch.stack(idx_rows).reshape(shape).to(torch.int32)
    return x, index


# 含 ties 的 pattern：indices 不做精确比对，仅校验 permutation 不变量。
_TIES_PATTERNS = {"all_zero", "all_same", "signed_zero_mix", "with_ties",
                  "nan_inf_combo", "inject_nan"}


def _exact_indices(pattern: str) -> bool:
    return pattern not in _TIES_PATTERNS


# ============================================================================
# 全量用例表（对齐 C++ ST GetFullCases）
#   字段：(case_id, dtype, shape, axis, descending, stable, pattern, is_empty, note)
# ============================================================================

def get_full_cases():
    cases = []
    dtypes = ["float16", "float32", "bfloat16", "int32"]

    # ---- A. 每 dtype：基础 shape × descending × stable ----
    for dt in dtypes:
        cases.append((dt, [8], -1, False, False, "random_unique", False, "1D 升序 unstable"))
        cases.append((dt, [8], -1, False, True, "random_unique", False, "1D 升序 stable"))
        cases.append((dt, [8], -1, True, False, "random_unique", False, "1D 降序 unstable"))
        cases.append((dt, [8], -1, True, True, "random_unique", False, "1D 降序 stable"))
        cases.append((dt, [4, 32], -1, False, True, "random_unique", False, "多行分核 升序"))
        cases.append((dt, [4, 32], -1, True, True, "random_unique", False, "多行分核 降序"))
        cases.append((dt, [3, 64], -1, False, True, "random_unique", False, "rank2 轴长64"))

    # ---- B. 多 rank（rank0–8）+ axis ----
    cases.append(("float16", [], -1, False, False, "scalar", False, "rank0 标量"))
    cases.append(("float32", [], -1, True, True, "scalar", False, "rank0 标量 降序"))
    cases.append(("int32", [], -1, False, False, "scalar", False, "rank0 标量 int32"))
    cases.append(("float16", [1], -1, False, True, "random_unique", False, "轴长=1 拷贝"))
    cases.append(("bfloat16", [1], -1, True, True, "random_unique", False, "轴长=1 拷贝 bf16"))
    cases.append(("float32", [4, 1], -1, False, True, "random_unique", False, "rank2 轴长=1"))
    cases.append(("int32", [4, 1], -1, True, True, "random_unique", False, "rank2 轴长=1 int32"))
    cases.append(("float16", [2, 3, 8], -1, False, False, "random_unique", False, "rank3"))
    cases.append(("bfloat16", [2, 2, 2, 16], -1, False, True, "random_unique", False, "rank4"))
    cases.append(("float16", [2, 2, 2, 2, 2, 2, 2, 4], -1, False, True, "random_unique", False, "rank8 axis=-1"))
    cases.append(("float32", [2, 2, 2, 2, 2, 2, 2, 4], 7, False, True, "random_unique", False, "rank8 axis=rank-1"))
    cases.append(("bfloat16", [3, 8], 1, True, True, "random_unique", False, "rank2 axis=rank-1 降序"))

    # ---- C. 单 tile 内多轴长（<= 单 tile 上限，不超界）----
    cases.append(("float16", [256], -1, False, True, "random_unique", False, "fp16 轴长256"))
    cases.append(("float16", [2048], -1, False, True, "random_unique", False, "fp16 轴长2048"))
    cases.append(("float16", [3008], -1, False, True, "random_unique", False, "fp16 轴长3008(近上限)"))
    cases.append(("float16", [3008], -1, True, True, "random_unique", False, "fp16 轴长3008 降序"))
    cases.append(("float32", [2048], -1, False, True, "random_unique", False, "fp32 轴长2048"))
    cases.append(("float32", [2816], -1, False, True, "random_unique", False, "fp32 轴长2816(近上限)"))
    cases.append(("bfloat16", [2048], -1, False, True, "random_unique", False, "bf16 轴长2048"))
    cases.append(("bfloat16", [2816], -1, False, True, "random_unique", False, "bf16 轴长2816(近上限)"))
    cases.append(("int32", [2048], -1, False, True, "random_unique", False, "int32 轴长2048"))
    cases.append(("int32", [2560], -1, False, True, "random_unique", False, "int32 轴长2560(近上限)"))
    cases.append(("float16", [16, 256], -1, False, True, "random_unique", False, "16行×256 多行较大轴"))

    # ---- D. 边界：空 tensor ----
    for dt in dtypes:
        cases.append((dt, [0], -1, False, False, "random_unique", True, "空 tensor [0]"))
    cases.append(("float16", [3, 0], -1, False, False, "random_unique", True, "空 tensor 多维 [3,0]"))
    cases.append(("float32", [0, 8], -1, False, False, "random_unique", True, "空 tensor 多维 [0,8]"))

    # ---- E. extreme（浮点 NaN/Inf/±0；int32 全零/全相等）----
    for dt in ["float16", "float32", "bfloat16"]:
        cases.append((dt, [8], -1, False, True, "inject_nan", False, "NaN 升序落开头"))
        cases.append((dt, [8], -1, True, True, "inject_nan", False, "NaN 降序落开头"))
        cases.append((dt, [8], -1, False, True, "single_pos_inf", False, "+Inf 升序末尾"))
        cases.append((dt, [8], -1, False, True, "single_neg_inf", False, "-Inf 升序开头"))
        cases.append((dt, [8], -1, False, True, "nan_inf_combo", False, "NaN+±Inf 综合"))
        cases.append((dt, [8], -1, False, True, "all_zero", False, "全零 ties"))
        cases.append((dt, [8], -1, False, True, "all_same", False, "全相等 ties stable"))
        cases.append((dt, [8], -1, False, True, "signed_zero_mix", False, "±0 视为相等 ties"))
    cases.append(("int32", [8], -1, False, True, "all_zero", False, "int32 全零 ties"))
    cases.append(("int32", [8], -1, False, True, "all_same", False, "int32 全相等 ties stable"))

    # ---- F. 确定性（含 ties 重复执行）----
    cases.append(("float16", [16], -1, False, True, "with_ties", False, "确定性 ties D1"))
    cases.append(("float32", [16, 64], -1, False, True, "with_ties", False, "确定性 多行分核 D2"))

    out = []
    for i, c in enumerate(cases):
        dt, shape, axis, desc, stable, pattern, is_empty, note = c
        out.append({
            "case_id": f"FULL-{dt}-{i + 1}",
            "dtype": dt, "shape": shape, "axis": axis,
            "descending": desc, "stable": stable, "pattern": pattern,
            "is_empty": is_empty, "note": note,
        })
    return out


# ---- 负向用例（仅 NPU 可用时构造；预期算子优雅拒绝）----
def get_negative_cases():
    # runtime_cap=True：拒绝阈值随实际可用 UB 运行时取值（以运行时拒绝为准，不写死）。
    # 此类用例在 PyTorch 路径作信息性核对（不硬判 pass/fail）。
    return [
        {"case_id": "N1-shape-mismatch", "dtype": "float16", "x_shape": [2, 3],
         "idx_shape": [2, 4], "axis": -1, "runtime_cap": False, "note": "x≠index shape → shape_mismatch"},
        {"case_id": "N2-axis-not-last", "dtype": "float16", "x_shape": [4, 8],
         "idx_shape": [4, 8], "axis": 0, "runtime_cap": False, "note": "axis=0 非最后一维"},
        {"case_id": "N3-axis-oob", "dtype": "float16", "x_shape": [4, 8],
         "idx_shape": [4, 8], "axis": 5, "runtime_cap": False, "note": "axis=5 越界"},
        {"case_id": "N4-axis-too-long", "dtype": "float16", "x_shape": [8192],
         "idx_shape": [8192], "axis": -1, "runtime_cap": True,
         "note": "fp16 N=8192 超单 tile 上限 → 运行时拒绝(阈值随 UB)"},
    ]


# ============================================================================
# Mock 比对闭环（无 NPU）：golden 充当实际输出，验证 compare.py 逻辑闭环。
#   对齐 C++ ST RunCaseMock —— 验证 golden + bitwise/isnan/不变量比对链路。
# ============================================================================

def run_case_mock(tc) -> bool:
    x, index = build_case_inputs(tc["dtype"], tc["shape"], tc["pattern"])
    golden_y, golden_si = compute_golden_sort_with_index(
        x, index, tc["axis"], tc["descending"], tc["stable"])

    if tc["is_empty"]:
        ok = (golden_y.numel() == 0) and (golden_si.numel() == 0)
        print(f"    {'[PASS]' if ok else '[FAIL]'} 空 tensor 返回空 "
              f"(y={golden_y.numel()} si={golden_si.numel()})")
        return ok

    # mock 实际输出 = golden（验证比对/不变量逻辑闭环）。
    return compare_results(golden_y, golden_si, golden_y.clone(), golden_si.clone(),
                           x, index, tc["descending"], _exact_indices(tc["pattern"]))


# ============================================================================
# Real 比对（NPU 可用时）：调用 torch.ops.sort_with_index.forward，取回 CPU 比对。
# ============================================================================

def run_case_real(tc, npu_ok) -> bool:
    x_cpu, index_cpu = build_case_inputs(tc["dtype"], tc["shape"], tc["pattern"])
    golden_y, golden_si = compute_golden_sort_with_index(
        x_cpu, index_cpu, tc["axis"], tc["descending"], tc["stable"])

    if not npu_ok:
        # torch_npu 不可用 → 跳过 NPU 实跑，回退 mock 闭环（仍验证 golden/比对逻辑）。
        return run_case_mock(tc)

    x = x_cpu.to("npu")
    index = index_cpu.to("npu")
    y, sorted_index = torch.ops.sort_with_index.forward(
        x, index, tc["axis"], tc["descending"], tc["stable"])
    y_cpu = y.cpu()
    si_cpu = sorted_index.cpu()

    if tc["is_empty"]:
        ok = (y_cpu.numel() == 0) and (si_cpu.numel() == 0)
        print(f"    {'[PASS]' if ok else '[FAIL]'} 空 tensor 返回空")
        return ok

    return compare_results(golden_y, golden_si, y_cpu, si_cpu,
                           x_cpu, index_cpu, tc["descending"], _exact_indices(tc["pattern"]))


def run_negative_real(nc, npu_ok) -> bool:
    if not npu_ok:
        print(f"    [SKIP] 负向用例 {nc['case_id']}：torch_npu 不可用")
        return None
    tdt = DTYPE_MAP[nc["dtype"]]
    x = torch.zeros(nc["x_shape"], dtype=tdt, device="npu")
    index = torch.zeros(nc["idx_shape"], dtype=torch.int32, device="npu")
    runtime_cap = nc.get("runtime_cap", False)
    try:
        torch.ops.sort_with_index.forward(x, index, nc["axis"], False, False)
        torch.npu.synchronize()
        if runtime_cap:
            # 运行时阈值类：未在本环境 UB 下拒绝 → 信息性核对，不硬判失败。
            print(f"    [INFO] 负向用例 {nc['case_id']} 本环境 UB 下未拒绝"
                  f"（阈值随 UB 运行时取值）")
            return None
        print(f"    [FAIL] 负向用例 {nc['case_id']} 未被拒绝")
        return False
    except Exception as e:
        print(f"    [PASS] 负向用例 {nc['case_id']} 被优雅拒绝：{type(e).__name__}")
        return True


# ============================================================================
# 主函数
# ============================================================================

def detect_npu():
    """检测 torch_npu + NPU 真实可用（CPU 版 torch 即便 import torch_npu 也无法执行）。"""
    try:
        import torch_npu
        return bool(getattr(torch, "npu", None)) and torch.npu.is_available()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="SortWithIndex ST —— ACLNN 两段式 PyTorch 接入")
    parser.add_argument("--lib", default=None, help="共享库路径 (libtorch_adapter.so)；不传则仅 mock 闭环")
    parser.add_argument("--case", type=int, default=None, help="执行指定用例编号 (0-based)")
    args = parser.parse_args()

    print("=" * 64)
    print("SortWithIndex ST（4 组 dtype + 边界 + extreme，910B NaN-开头语义）")
    print("=" * 64)

    lib_loaded = False
    if args.lib:
        lib = os.path.realpath(args.lib)
        if not os.path.exists(lib):
            print(f"错误: 共享库不存在 {lib}")
            return 1
        torch.ops.load_library(lib)
        lib_loaded = True
        print(f"已加载共享库: {lib}")
    # Real NPU 执行需同时满足：torch_npu+NPU 可用 且 已加载 torch_adapter（注册 forward op）。
    npu_ok = detect_npu() and lib_loaded
    if npu_ok:
        print("模式: Real (torch_npu + NPU 实跑 + bitwise(isnan-aware)/精确/不变量比对)")
    else:
        print("模式: Mock 闭环 (torch_npu 不可用或未传 --lib)")
        print("     验证 golden(910B NaN-开头) + bitwise/isnan/不变量比对链路闭环。")

    # ---- CPU Golden 自测（始终运行）----
    print("\n" + "=" * 64)
    print("CPU Golden 正确性自测（910B NaN-开头）")
    print("=" * 64)
    golden_ok = test_golden_correctness()
    print(f"\nGolden 自测: {'PASS' if golden_ok else 'FAIL'}")

    # ---- 全量用例 ----
    cases = get_full_cases()
    if args.case is not None:
        if not (0 <= args.case < len(cases)):
            print(f"错误: 用例编号超出范围 (0-{len(cases) - 1})")
            return 1
        cases = [cases[args.case]]

    print("\n" + "=" * 64)
    print(f"执行全量用例（共 {len(cases)} 条）")
    print("=" * 64)
    passed = failed = 0
    for tc in cases:
        print(f"\n用例 {tc['case_id']} [{tc['dtype']}]: shape={tc['shape']} axis={tc['axis']} "
              f"desc={int(tc['descending'])} stable={int(tc['stable'])} ({tc['note']})")
        ok = run_case_real(tc, npu_ok)
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  用例 {tc['case_id']}: {'PASS' if ok else 'FAIL'}")

    # ---- 负向用例 ----
    print("\n" + "=" * 64)
    print("负向用例（预期算子优雅拒绝）")
    print("=" * 64)
    neg_skipped = 0
    for nc in get_negative_cases():
        print(f"\n负向 {nc['case_id']}: x={nc['x_shape']} idx={nc['idx_shape']} axis={nc['axis']} ({nc['note']})")
        r = run_negative_real(nc, npu_ok)
        if r is None:
            neg_skipped += 1
        elif r:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 64)
    print("测试报告")
    print("=" * 64)
    print(f"Golden 自测: {'PASS' if golden_ok else 'FAIL'}")
    print(f"用例通过: {passed}")
    print(f"用例失败: {failed}")
    print(f"负向跳过(torch_npu 不可用): {neg_skipped}")
    if not npu_ok:
        print("NPU 实链: SKIP（torch_npu 不可用）")
    print("=" * 64)

    return 0 if (golden_ok and failed == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
