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
精度比对模块 —— SortWithIndex（重排算子，全 dtype rtol=atol=0）

================================================================================
SortWithIndex 是「索引跟随排序」重排算子：y/sorted_index 元素仅搬移、未参与算术 → **精确比对**（非浮点容差，二进制一致）。

比对口径（对齐 C++ ST CompareValuesIsnanAware / CompareIndicesExact / CheckInvariants910b）：
  - y（values）   ：bitwise 精确（有限值 memcmp 等价）；NaN 行按 isnan（升序 Muls(-1) 翻 NaN
                    符号位，bit 不一致但仍是 NaN → NaN↔NaN 视为相等，不计 bitwise 失配）。
  - sorted_index ：整型精确相等（rtol=atol=0）。仅对「无 ties 输入」启用精确比对；含 ties 用例
                    （ALL_ZERO/ALL_SAME/±0/NaN/Inf 等）只校验 permutation 不变量。
  - 结构性不变量：
      * y_is_permutation_of_x（multiset，NaN↔NaN isnan 等价）
      * sorted_index_is_permutation_of_index
      * y 沿最后一维单调（升序不减/降序不增；NaN 排除在单调性之外，落开头）。
================================================================================
"""

import numpy as np
import torch


# ============================================================================
# 工具：value tensor -> 排序键 float（bf16/int32 经 float）+ isnan 掩码 + bitwise 位型容器
# ============================================================================

def _to_key_np(t: torch.Tensor) -> np.ndarray:
    """value -> float64 排序键（序关系比对 / isnan 判定用）。"""
    if t.dtype == torch.bfloat16:
        return t.detach().cpu().to(torch.float32).numpy().astype(np.float64)
    return t.detach().cpu().to(torch.float64).numpy()


def _to_bits_np(t: torch.Tensor) -> np.ndarray:
    """value -> 原始位型整型容器（bitwise 精确比对用）。"""
    if t.dtype == torch.bfloat16:
        return t.detach().cpu().view(torch.int16).numpy()
    if t.dtype == torch.float16:
        return t.detach().cpu().view(torch.int16).numpy()
    if t.dtype == torch.float32:
        return t.detach().cpu().view(torch.int32).numpy()
    # int32 / 其余整型：值本身即位型。
    return t.detach().cpu().numpy()


# ============================================================================
# value 比对：bitwise（有限值）+ isnan-aware（NaN 行）
# ============================================================================

def compare_values_isnan_aware(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """values bitwise(isnan-aware) 比对。NaN 行：两侧均 NaN 视为相等；有限值：位型精确。"""
    if golden.shape != actual.shape:
        print(f"    [FAIL] values shape 不一致: {tuple(golden.shape)} vs {tuple(actual.shape)}")
        return False
    g_key = _to_key_np(golden).reshape(-1)
    a_key = _to_key_np(actual).reshape(-1)
    g_bits = _to_bits_np(golden).reshape(-1)
    a_bits = _to_bits_np(actual).reshape(-1)

    g_nan = np.isnan(g_key)
    a_nan = np.isnan(a_key)
    nan_row = g_nan | a_nan

    # NaN 行：两侧均 NaN 视为相等。
    nan_ok = (g_nan == a_nan) | (g_nan & a_nan)  # 等价：两侧 isnan 一致；进一步要求 NaN 行均 NaN
    nan_eq = ~nan_row | (g_nan & a_nan)
    # 有限值行：位型精确。
    finite_eq = (g_bits == a_bits)
    eq = np.where(nan_row, nan_eq, finite_eq)

    mismatch = int((~eq).sum())
    if mismatch == 0:
        print(f"    [PASS] values bitwise(isnan-aware) 一致（{eq.size} 个元素）")
        return True
    bad = np.flatnonzero(~eq)
    for i in bad[:5]:
        print(f"    values 不匹配 [{i}]: golden_key={g_key[i]:.6f}(nan={int(g_nan[i])}) "
              f"actual_key={a_key[i]:.6f}(nan={int(a_nan[i])})")
    print(f"    [FAIL] values 发现 {mismatch} 个不匹配")
    return False


# ============================================================================
# indices 比对：整型精确
# ============================================================================

def compare_indices_exact(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """sorted_index 精确比对（rtol=atol=0）。"""
    if golden.shape != actual.shape:
        print(f"    [FAIL] indices shape 不一致: {tuple(golden.shape)} vs {tuple(actual.shape)}")
        return False
    g = golden.detach().cpu().numpy().reshape(-1)
    a = actual.detach().cpu().numpy().reshape(-1)
    eq = (g == a)
    mismatch = int((~eq).sum())
    if mismatch == 0:
        print(f"    [PASS] indices 精确一致（{eq.size} 个元素）")
        return True
    bad = np.flatnonzero(~eq)
    for i in bad[:5]:
        print(f"    indices 不匹配 [{i}]: golden={g[i]} actual={a[i]}")
    print(f"    [FAIL] indices 发现 {mismatch} 个不匹配")
    return False


# ============================================================================
# 结构性不变量（置换性 ×2 + 单调性）
# ============================================================================

def _multiset_key(key_row: np.ndarray, bits_row: np.ndarray) -> np.ndarray:
    """NaN-aware multiset 比较 key：NaN 归一为同一 canonical key；有限值用原始位型。"""
    out = bits_row.astype(np.int64).copy()
    out[np.isnan(key_row)] = np.iinfo(np.int64).max  # 所有 NaN 归一
    return out


def check_invariants(x: torch.Tensor, index: torch.Tensor, y: torch.Tensor,
                     sorted_index: torch.Tensor, descending: bool) -> bool:
    """3 个结构性不变量（置换性 ×2 + 单调性）。NaN 行 isnan 等价、排除在单调性之外。"""
    if x.numel() == 0:
        ok = (x.shape == y.shape) and (index.shape == sorted_index.shape)
        if not ok:
            print("    [FAIL] 空 tensor 输出 shape 不一致")
        return ok

    # rank0 标量：单元素平凡置换，y==x、sorted_index==index 即成立。
    if x.dim() == 0:
        ok = compare_values_isnan_aware(x, y) and compare_indices_exact(index, sorted_index)
        if ok:
            print("    [PASS] rank0 标量置换不变量成立")
        return ok

    axis_len = x.shape[-1]
    x_key = _to_key_np(x).reshape(-1, axis_len)
    y_key = _to_key_np(y).reshape(-1, axis_len)
    x_bits = _to_bits_np(x).reshape(-1, axis_len)
    y_bits = _to_bits_np(y).reshape(-1, axis_len)
    idx_np = index.detach().cpu().numpy().reshape(-1, axis_len)
    si_np = sorted_index.detach().cpu().numpy().reshape(-1, axis_len)

    rows = x_key.shape[0]
    for r in range(rows):
        # value multiset（NaN↔NaN isnan 等价）
        xa = np.sort(_multiset_key(x_key[r], x_bits[r]))
        ya = np.sort(_multiset_key(y_key[r], y_bits[r]))
        if not np.array_equal(xa, ya):
            print(f"    [FAIL] 不变量 y_is_permutation_of_x 失败（row={r}）")
            return False
        # index multiset
        if not np.array_equal(np.sort(idx_np[r]), np.sort(si_np[r])):
            print(f"    [FAIL] 不变量 sorted_index_is_permutation_of_index 失败（row={r}）")
            return False
        # 单调：跳过 NaN（NaN 落开头，不计单调）
        finite = y_key[r][~np.isnan(y_key[r])]
        if finite.size >= 2:
            diffs = np.diff(finite)
            bad = (diffs < 0).any() if not descending else (diffs > 0).any()
            if bad:
                print(f"    [FAIL] 不变量 y_sorted_monotone 失败（row={r}）")
                return False
    print("    [PASS] 3 个结构性不变量成立")
    return True


# ============================================================================
# 统一入口
# ============================================================================

def compare_results(golden_y: torch.Tensor, golden_si: torch.Tensor,
                    actual_y: torch.Tensor, actual_si: torch.Tensor,
                    x: torch.Tensor, index: torch.Tensor,
                    descending: bool, exact_indices: bool) -> bool:
    """
    SortWithIndex 综合比对：value bitwise(isnan-aware) + index 精确（无 ties 时）+ 3 个不变量。

    Args:
        golden_y/golden_si : CPU golden 输出
        actual_y/actual_si : NPU（或 mock）实际输出，已取回 CPU
        x/index            : 原始输入（不变量校验用）
        descending         : 排序方向
        exact_indices      : 是否对 indices 做精确比对（仅无 ties 输入 True）
    """
    val_ok = compare_values_isnan_aware(golden_y, actual_y)
    if exact_indices:
        idx_ok = compare_indices_exact(golden_si, actual_si)
    else:
        print("    [INFO] ties 用例：跳过 indices 精确比对，仅断言 permutation 不变量")
        idx_ok = True
    inv_ok = check_invariants(x, index, actual_y, actual_si, descending)
    return val_ok and idx_ok and inv_ok
