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
ST 测试 - FusedMulAddN ACLNN 两段式接口 PyTorch 接入验证（L0 + L1 全量）

算子公式: y_i = x1_i * x3[0] + x2_i  （x3 单元素标量张量，仅取 x3[0]）

数据流：
    PyTorch Tensor (NPU) -> torch_adapter.so -> aclnnFusedMulAddN -> NPU 结果
    CPU golden (golden.py)  <-compare.py->  NPU 实际结果（取回 CPU 后比对）

用例覆盖（对应 C++ ST 全量，L0_001~005 / L1_001~046）：
    - 5 dtype: float32/float16/bfloat16/int32/int16
    - 多 shape: rank0/[1]/[1,1]/1D/2D/3D/4D/大 shape(4096x1024)
    - 边界不变量: x3=0 ⇒ y==x2; x3=1 ⇒ y==x1+x2; 空 tensor returns_empty
    - 极端输入: NaN 传播 / +Inf / 全零 / fp16 上界 / 整数上下界回绕
    - 广播: x3 形态 [1,1] 等价单元素标量
    - 确定性: 同输入连续执行 3 次 bitwise 一致

使用方法：
    python3 test.py --lib /path/to/libtorch_adapter.so
    python3 test.py --lib libtorch_adapter.so --case L1_023
    python3 test.py --golden-only          # 仅 CPU golden 自测（无需 NPU）
    python3 test.py --lib ... --level L0    # 只跑 L0
"""

import argparse
import os
import sys
import time

import torch

from golden import compute_golden_fused_mul_add_n, test_golden_correctness
from compare import compare_results

CASE_COOLDOWN_MS = int(os.environ.get("CASE_COOLDOWN_MS", 50))

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int16": torch.int16,
}
INT_DTYPES = (torch.int32, torch.int16)


# ============================================================================
# 输入数据合成（确定性：固定 seed，便于复现 / determinism 重复执行）
# ============================================================================

def _make_tensor(shape, dtype, pattern, seed):
    """根据 pattern 合成 CPU 输入 tensor。"""
    g = torch.Generator().manual_seed(seed)

    if pattern == "empty":
        return torch.zeros(shape, dtype=dtype)

    if pattern == "all_zero":
        return torch.zeros(shape, dtype=dtype)

    if isinstance(pattern, str) and pattern.startswith("all_same:"):
        val = float(pattern.split(":", 1)[1])
        if dtype in INT_DTYPES:
            return torch.full(shape, int(val), dtype=dtype)
        return torch.full(shape, val, dtype=dtype)

    if pattern == "pos_inf":
        t = torch.full(shape, float("inf"), dtype=dtype)
        return t

    if pattern == "inject_nan_one":
        # 随机基底 + 首元素置 NaN
        t = (torch.rand(shape, generator=g, dtype=torch.float32) * 4 - 2).to(dtype)
        flat = t.reshape(-1)
        if flat.numel() > 0:
            flat[0] = float("nan")
        return flat.reshape(shape)

    if pattern in ("random", "random_int"):
        if dtype in INT_DTYPES:
            # 落在 [-32, 32)，避免常规用例直算溢出（溢出语义由专门用例覆盖）
            t = torch.randint(-32, 32, shape if shape != () else (1,), generator=g, dtype=torch.int64)
            t = t.to(dtype)
            return t.reshape(shape) if shape != () else t.reshape(())
        # 浮点随机 [-2, 2)
        if shape == ():
            return (torch.rand((), generator=g, dtype=torch.float32) * 4 - 2).to(dtype)
        return (torch.rand(shape, generator=g, dtype=torch.float32) * 4 - 2).to(dtype)

    raise ValueError(f"未知 pattern: {pattern}")


def _make_x3(x3_shape, dtype, x3_value):
    """x3：单元素标量张量。x3_shape 形态可为 (), (1,), (1,1)。"""
    if dtype in INT_DTYPES:
        return torch.full(x3_shape, int(x3_value), dtype=dtype)
    return torch.full(x3_shape, float(x3_value), dtype=dtype)


# ============================================================================
# 用例定义
#   每条用例字段:
#     id, level, dtype, shape (x1/x2/y), x3_shape, x3_value,
#     x1_pattern, x2_pattern, check ("oracle"|"invariant_x2"|"empty"|"nan"|"deterministic"),
#     seed
# ============================================================================

def _c(cid, level, dtype, shape, x3_value, x1p="random", x2p="random",
       x3_shape=(1,), check="oracle", seed=1234):
    return {
        "id": cid, "level": level, "dtype": DTYPE_MAP[dtype], "dtype_name": dtype,
        "shape": shape, "x3_shape": x3_shape, "x3_value": x3_value,
        "x1_pattern": x1p, "x2_pattern": x2p, "check": check, "seed": seed,
    }


def get_test_cases():
    cases = []
    # ---------------- L0 门槛用例（L0_001~005）----------------
    cases += [
        _c("L0_001", "L0", "float32", (2, 3), 2.0, seed=101),
        _c("L0_002", "L0", "float32", (8,), 1.5, seed=102),
        _c("L0_003", "L0", "float16", (2, 3), 2.0, seed=103),
        _c("L0_004", "L0", "int32", (2, 3), 3, "random_int", "random_int", seed=104),
        _c("L0_005", "L0", "float32", (1,), 1.0, seed=105),
    ]

    # ---------------- L1: 5 dtype × 多 shape（L1_001~016）----------------
    cases += [
        _c("L1_001", "L1", "float32", (2, 3), 2.0, seed=201),
        _c("L1_002", "L1", "float32", (32,), -1.5, seed=202),
        _c("L1_003", "L1", "float32", (2, 3, 4), 0.5, seed=203),
        _c("L1_004", "L1", "float32", (2, 3, 4, 5), 3.0, seed=204),
        _c("L1_005", "L1", "float16", (2, 3), 2.0, seed=205),
        _c("L1_006", "L1", "float16", (32,), -2.5, seed=206),
        _c("L1_007", "L1", "float16", (2, 3, 4), 1.5, seed=207),
        _c("L1_008", "L1", "bfloat16", (2, 3), 2.0, seed=208),
        _c("L1_009", "L1", "bfloat16", (32,), -1.0, seed=209),
        _c("L1_010", "L1", "bfloat16", (2, 3, 4), 0.75, seed=210),
        _c("L1_011", "L1", "int32", (2, 3), 3, "random_int", "random_int", seed=211),
        _c("L1_012", "L1", "int32", (32,), -2, "random_int", "random_int", seed=212),
        _c("L1_013", "L1", "int32", (2, 3, 4), 5, "random_int", "random_int", seed=213),
        _c("L1_014", "L1", "int16", (2, 3), 3, "random_int", "random_int", seed=214),
        _c("L1_015", "L1", "int16", (32,), -1, "random_int", "random_int", seed=215),
        _c("L1_016", "L1", "int16", (2, 3, 4), 2, "random_int", "random_int", seed=216),
    ]

    # ---------------- L1: rank=0 / 单元素 / x3形态1x1 / 大 shape（L1_017~025）----------------
    cases += [
        _c("L1_017", "L1", "float32", (), 2.0, seed=217),
        _c("L1_018", "L1", "float16", (), 2.0, seed=218),
        _c("L1_019", "L1", "int32", (), 3, "random_int", "random_int", seed=219),
        _c("L1_020", "L1", "float32", (1,), 1.0, seed=220),
        _c("L1_021", "L1", "bfloat16", (1,), 2.0, seed=221),
        _c("L1_022", "L1", "float32", (1, 1), 2.0, x3_shape=(1, 1), seed=222),  # x3 形态 [1,1]
        _c("L1_023", "L1", "float32", (4096, 1024), 2.0, seed=223),
        _c("L1_024", "L1", "float16", (4096, 1024), 2.0, seed=224),
        _c("L1_025", "L1", "int32", (4096, 1024), 2, "random_int", "random_int", seed=225),
    ]

    # ---------------- L1: x3=0 不变量 / x3=1 退化（L1_026~030）----------------
    cases += [
        _c("L1_026", "L1", "float32", (4, 5), 0.0, check="invariant_x2", seed=226),
        _c("L1_027", "L1", "float16", (4, 5), 0.0, check="invariant_x2", seed=227),
        _c("L1_028", "L1", "int32", (4, 5), 0, "random_int", "random_int", check="invariant_x2", seed=228),
        _c("L1_029", "L1", "float32", (4, 5), 1.0, seed=229),  # x3=1 ⇒ y==x1+x2 (oracle)
        _c("L1_030", "L1", "bfloat16", (4, 5), 1.0, seed=230),
    ]

    # ---------------- L1: 空 tensor returns_empty（L1_031~032）----------------
    cases += [
        _c("L1_031", "L1", "float32", (0, 3), 2.0, "empty", "empty", check="empty", seed=231),
        _c("L1_032", "L1", "int32", (0, 3), 2, "empty", "empty", check="empty", seed=232),
    ]

    # ---------------- L1: 极端输入 NaN/Inf/全零/上界（L1_033~040）----------------
    cases += [
        _c("L1_033", "L1", "float32", (8,), 2.0, "inject_nan_one", "random", check="nan", seed=233),
        _c("L1_034", "L1", "float16", (8,), 2.0, "inject_nan_one", "random", check="nan", seed=234),
        _c("L1_035", "L1", "bfloat16", (8,), 2.0, "inject_nan_one", "random", check="nan", seed=235),
        _c("L1_036", "L1", "float32", (1,), 2.0, "pos_inf", "random", seed=236),
        _c("L1_037", "L1", "float16", (1,), 2.0, "pos_inf", "random", seed=237),
        _c("L1_038", "L1", "float32", (8,), 0.0, "all_zero", "all_zero", seed=238),
        _c("L1_039", "L1", "int32", (8,), 0, "all_zero", "all_zero", seed=239),
        _c("L1_040", "L1", "float16", (1,), 1.0, "all_same:60000.0", "all_same:60000.0", seed=240),
    ]

    # ---------------- L1: 整数上下界回绕（L1_041~044）----------------
    cases += [
        _c("L1_041", "L1", "int32", (1,), 1, "all_same:2147483647", "all_same:1", seed=241),
        _c("L1_042", "L1", "int16", (1,), 1, "all_same:32767", "all_same:1", seed=242),
        _c("L1_043", "L1", "int32", (8,), 1, "all_same:-2147483648", "all_zero", seed=243),
        _c("L1_044", "L1", "int16", (8,), 1, "all_same:-32768", "all_zero", seed=244),
    ]

    # ---------------- L1: 确定性（L1_045~046）----------------
    cases += [
        _c("L1_045", "L1", "float32", (2, 3), 2.0, check="deterministic", seed=245),
        _c("L1_046", "L1", "float16", (4096, 1024), 2.0, check="deterministic", seed=246),
    ]

    return cases


# ============================================================================
# 单条 NPU 执行：构造输入 -> 算子 -> 取回结果
# ============================================================================

def _build_inputs_cpu(tc):
    x1 = _make_tensor(tc["shape"], tc["dtype"], tc["x1_pattern"], tc["seed"])
    x2 = _make_tensor(tc["shape"], tc["dtype"], tc["x2_pattern"], tc["seed"] + 7)
    x3 = _make_x3(tc["x3_shape"], tc["dtype"], tc["x3_value"])
    return x1, x2, x3


def _run_op_npu(x1_cpu, x2_cpu, x3_cpu):
    x1 = x1_cpu.to("npu")
    x2 = x2_cpu.to("npu")
    x3 = x3_cpu.to("npu")
    result = torch.ops.fused_mul_add_n.forward(x1, x2, x3)
    return result.cpu()


def run_test(tc):
    print(f"\n[{tc['id']}] {tc['dtype_name']} shape={tuple(tc['shape'])} "
          f"x3={tc['x3_value']} check={tc['check']}")

    x1_cpu, x2_cpu, x3_cpu = _build_inputs_cpu(tc)
    assert tuple(x1_cpu.shape) == tuple(tc["shape"]) and tuple(x2_cpu.shape) == tuple(tc["shape"]), \
        f"{tc['id']}: 输入 shape 不匹配"
    assert x3_cpu.numel() == 1, f"{tc['id']}: x3 必须单元素"

    golden = compute_golden_fused_mul_add_n(x1_cpu, x2_cpu, x3_cpu)

    if not torch.npu.is_available():
        print("  [SKIP] NPU 不可用")
        return None

    actual = _run_op_npu(x1_cpu, x2_cpu, x3_cpu)

    passed = True
    check = tc["check"]

    if check == "empty":
        # returns_empty: 形状一致且元素数为 0
        shape_ok = tuple(actual.shape) == tuple(tc["shape"])
        empty_ok = actual.numel() == 0
        passed = shape_ok and empty_ok
        print(f"  returns_empty: shape={tuple(actual.shape)} numel={actual.numel()} "
              f"{'PASS' if passed else 'FAIL'}")

    elif check == "invariant_x2":
        # x3=0 ⇒ y == x2（zero_multiplier_yields_x2）。先验证 golden 满足不变量，再对拍 NPU。
        inv_golden = bool(
            torch.equal(golden, x2_cpu) if tc["dtype"] in INT_DTYPES
            else torch.allclose(golden.float(), x2_cpu.float(), rtol=1e-6, atol=1e-6))
        if not inv_golden:
            print("  [FAIL] golden 未满足不变量 y==x2")
        passed = inv_golden and compare_results(golden, actual)

    elif check == "nan":
        # NaN 传播：NaN 位置一致 + 非 NaN 位置数值一致（compare_results 已含特殊位置判定）
        g_nan = torch.isnan(golden.float())
        a_nan = torch.isnan(actual.float())
        nan_pos_ok = bool(torch.equal(g_nan, a_nan)) and bool(g_nan.any())
        if not nan_pos_ok:
            print(f"  [FAIL] NaN 位置不一致 golden_nan={g_nan.tolist()} actual_nan={a_nan.tolist()}")
        passed = nan_pos_ok and compare_results(golden, actual)

    elif check == "deterministic":
        # 同输入连续执行 3 次，结果 bitwise 一致，且与 golden 对拍通过
        runs = [actual]
        for _ in range(2):
            runs.append(_run_op_npu(x1_cpu, x2_cpu, x3_cpu))
        det_ok = all(torch.equal(runs[0], r) for r in runs[1:])
        print(f"  determinism(3x bitwise): {'PASS' if det_ok else 'FAIL'}")
        passed = det_ok and compare_results(golden, runs[0])

    else:  # "oracle"
        passed = compare_results(golden, actual)

    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    if CASE_COOLDOWN_MS > 0:
        time.sleep(CASE_COOLDOWN_MS / 1000.0)
    return passed


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ST 测试 - FusedMulAddN ACLNN 两段式 PyTorch 接入")
    parser.add_argument("--lib", default=None, help="共享库路径 (libtorch_adapter.so)")
    parser.add_argument("--case", type=str, default=None, help="执行指定用例 id（如 L1_023）")
    parser.add_argument("--level", type=str, default=None, choices=["L0", "L1"], help="只跑指定级别")
    parser.add_argument("--golden-only", action="store_true", help="仅 CPU golden 自测，不上 NPU")
    args = parser.parse_args()

    print("=" * 64)
    print("CPU Golden 自测 (y = x1*x3[0] + x2)")
    print("=" * 64)
    golden_passed = test_golden_correctness()
    print(f"\nGolden 自测: {'PASS' if golden_passed else 'FAIL'}")
    if not golden_passed:
        print("Golden 自测失败，终止。")
        return 1

    if args.golden_only:
        print("\n[--golden-only] 跳过 NPU 测试。")
        return 0

    if not args.lib:
        print("\n错误: 非 --golden-only 模式需提供 --lib")
        return 2
    lib = os.path.realpath(args.lib)
    if not os.path.exists(lib):
        print(f"错误: 文件不存在 {lib}")
        return 2
    print(f"\n加载共享库: {lib}")
    try:
        import torch_npu  # noqa: F401  确保 NPU 后端注册
    except Exception as e:  # pragma: no cover
        print(f"警告: 导入 torch_npu 失败: {e}")
    torch.ops.load_library(lib)
    print("共享库加载成功")

    test_cases = get_test_cases()
    if args.level:
        test_cases = [tc for tc in test_cases if tc["level"] == args.level]
    if args.case:
        test_cases = [tc for tc in test_cases if tc["id"] == args.case]
        if not test_cases:
            print(f"错误: 未找到用例 {args.case}")
            return 2

    print(f"\n{'=' * 64}")
    print(f"设备: NPU | 用例数: {len(test_cases)}")
    print(f"{'=' * 64}")

    passed_count = failed_count = skipped_count = 0
    failed_ids = []
    for tc in test_cases:
        r = run_test(tc)
        if r is None:
            skipped_count += 1
        elif r:
            passed_count += 1
        else:
            failed_count += 1
            failed_ids.append(tc["id"])

    print(f"\n{'=' * 64}")
    print("总体测试报告")
    print(f"{'=' * 64}")
    print(f"总计: {len(test_cases)}  通过: {passed_count}  失败: {failed_count}  跳过: {skipped_count}")
    if failed_ids:
        print("失败用例:")
        for cid in failed_ids:
            print(f"  - {cid}")
    print(f"{'=' * 64}")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
