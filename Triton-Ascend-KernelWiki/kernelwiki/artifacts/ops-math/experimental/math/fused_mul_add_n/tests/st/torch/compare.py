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
精度比对模块 - FusedMulAddN

对标 CANN 算子精度验收【社区标准】：
- 浮点计算类（float32/float16/bfloat16）：MERE/MARE Threshold 方法
    通过条件：MERE < Threshold 且 MARE < 10 * Threshold
- 整数计算类（int32/int16）：bitwise_equal（二进制一致 / 绝对误差 0）

阈值（社区标准 per_dtype）：
    float32 = 2^-13 ≈ 1.220703125e-04
    float16 = 2^-10 ≈ 9.765625e-04
    bfloat16= 2^-7  ≈ 7.8125e-03
    int32/int16 = 0 (bitwise)

特殊位置处理（与极端用例语义对齐）：
  - golden 或 actual 中 NaN/Inf 位置不参与相对误差统计；改为按位置一致性判定
    （NaN<->NaN、+Inf<->+Inf、-Inf<->-Inf 视为匹配）。
"""

import torch

# ============================================================================
# 浮点 MERE 阈值（社区标准 per_dtype）
# ============================================================================

FLOAT_THRESHOLDS = {
    torch.float16: 9.765625e-04,    # 2^-10
    torch.bfloat16: 7.8125e-03,     # 2^-7
    torch.float32: 1.220703125e-04,  # 2^-13
}

_INT_DTYPES = (torch.int32, torch.int16)


def get_threshold(dtype: torch.dtype) -> float:
    """根据 dtype 获取 MERE 阈值（per_dtype）。"""
    return FLOAT_THRESHOLDS[dtype]


# ============================================================================
# 浮点比对：MERE/MARE Threshold（NaN/Inf 位置单独判定）
# ============================================================================

def _compare_float(golden: torch.Tensor, actual: torch.Tensor, dtype: torch.dtype) -> bool:
    golden_f = golden.flatten().float()
    actual_f = actual.flatten().float()

    if golden_f.numel() == 0:
        # 空 tensor：形状一致即视为通过（returns_empty 由 test.py 额外断言 shape）
        print("    空 tensor，跳过数值比对（形状一致即通过）")
        return True

    # 特殊位置（NaN/Inf）掩码
    special = (torch.isnan(golden_f) | torch.isinf(golden_f) |
               torch.isnan(actual_f) | torch.isinf(actual_f))

    special_pass = True
    if special.any():
        gs, as_ = golden_f[special], actual_f[special]
        # NaN<->NaN 匹配
        nan_match = torch.isnan(gs) & torch.isnan(as_)
        # +Inf<->+Inf, -Inf<->-Inf 匹配（同号无穷）
        inf_match = torch.isinf(gs) & torch.isinf(as_) & (torch.sign(gs) == torch.sign(as_))
        pos_ok = nan_match | inf_match
        special_pass = bool(pos_ok.all())
        if not special_pass:
            bad = (~pos_ok).nonzero().flatten()
            first = bad[0].item()
            print(f"    特殊位置(NaN/Inf)不匹配数: {bad.numel()} "
                  f"(首个: golden={gs[first].item()}, actual={as_[first].item()})")

    # 常规位置统计
    normal = ~special
    if normal.any():
        gn, an = golden_f[normal], actual_f[normal]
        eps = torch.finfo(torch.float32).tiny
        rel = (gn - an).abs() / (gn.abs() + eps)
        mere = rel.mean().item()
        mare = rel.max().item()
        threshold = get_threshold(dtype)
        mare_threshold = 10 * threshold
        mere_pass = mere < threshold
        mare_pass = mare < mare_threshold
        num_pass = mere_pass and mare_pass
        if not num_pass:
            print(f"    MERE: {mere:.6e} (阈值: {threshold:.6e}) {'PASS' if mere_pass else 'FAIL'}")
            print(f"    MARE: {mare:.6e} (阈值: {mare_threshold:.6e}) {'PASS' if mare_pass else 'FAIL'}")
            diff = (gn - an).abs()
            midx = diff.argmax().item()
            print(f"    最大差异位置: golden={gn[midx].item():.6f}, actual={an[midx].item():.6f}")
    else:
        num_pass = True

    return special_pass and num_pass


# ============================================================================
# 整型比对：bitwise_equal（绝对误差 0）
# ============================================================================

def _compare_integer(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    g = golden.flatten()
    a = actual.flatten().to(g.dtype)
    if g.numel() == 0:
        print("    空 tensor，跳过数值比对（形状一致即通过）")
        return True
    passed = bool(torch.equal(g, a))
    if not passed:
        mask = g != a
        mismatch = int(mask.sum().item())
        first_idx = int(mask.nonzero()[0].item())
        print(f"    不匹配元素数: {mismatch} (首个位置: {first_idx})")
        print(f"    Golden[{first_idx}] = {g[first_idx].item()}")
        print(f"    Actual[{first_idx}] = {a[first_idx].item()}")
    return passed


# ============================================================================
# 统一入口
# ============================================================================

def compare_results(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """比对 golden 与实际结果，按 dtype 选用浮点/整型标准；失败打印差异详情。"""
    # 形状一致性前置校验（含空 tensor 场景）
    if tuple(golden.shape) != tuple(actual.shape):
        print(f"    [FAIL] shape 不一致: golden={tuple(golden.shape)} vs actual={tuple(actual.shape)}")
        return False

    dtype = golden.dtype
    if dtype in _INT_DTYPES:
        return _compare_integer(golden, actual)
    return _compare_float(golden, actual, dtype)
