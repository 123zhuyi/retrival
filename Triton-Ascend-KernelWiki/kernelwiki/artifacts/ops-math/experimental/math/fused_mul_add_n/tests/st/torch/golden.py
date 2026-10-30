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
CPU Golden 计算模块 - FusedMulAddN

算子公式：
    y_i = x1_i * x3[0] + x2_i
其中 x3 是单元素标量张量（ShapeSize=1），仅取 x3[0] 作为标量乘数，按标量广播到 x1 全部元素。
reference_oracle：tmp = numpy.multiply(x1, x3); y = numpy.add(tmp, x2)

精度/语义约定（与 C++ ST golden 对齐）：
  - 浮点（float32/float16/bfloat16）：golden 在 float32 域计算（multiply -> add），
    末尾再 cast 回目标 dtype。对齐 kernel cast->fp32->cast 策略，避免低精度中间溢出引入额外误差。
  - 整型（int32/int16）：按目标整型【两步回绕】语义计算，不做饱和：
        tmp = wrap_T(x1 * x3[0])    # 第一步 mul，回绕到 T
        y   = wrap_T(tmp + x2)      # 第二步 add，回绕到 T
    与两步语义（mul -> add）和 kernel/硬件逐算子回绕一致。
"""

import numpy as np
import torch

# 整型 dtype -> numpy 整型（用于显式回绕语义计算）
_INT_NP = {
    torch.int32: np.int32,
    torch.int16: np.int16,
}

_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _is_int_dtype(dtype: torch.dtype) -> bool:
    return dtype in _INT_NP


def compute_golden_fused_mul_add_n(
    x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
) -> torch.Tensor:
    """
    计算 FusedMulAddN 的 CPU golden 结果: y = x1 * x3[0] + x2

    参数:
        x1: 主张量（任意 ND，含 rank=0 标量、空 tensor）
        x2: 加项，与 x1 同 shape 同 dtype
        x3: 单元素标量张量（ShapeSize=1），仅取首元素 x3[0]
    返回:
        与 x1 同 shape 同 dtype 的 golden tensor（CPU）
    """
    assert x1.shape == x2.shape, f"golden: x1/x2 shape 必须一致 {tuple(x1.shape)} vs {tuple(x2.shape)}"
    assert x1.dtype == x2.dtype == x3.dtype, (
        f"golden: x1/x2/x3 dtype 必须一致 {x1.dtype}/{x2.dtype}/{x3.dtype}"
    )
    assert x3.numel() == 1, f"golden: x3 必须为单元素标量张量 (ShapeSize=1)，实际 numel={x3.numel()}"

    dtype = x1.dtype

    # x3[0]：标量乘数（按标量广播）。reshape(-1)[0] 兼容 [], [1], [1,1] 等形态。
    scalar = x3.detach().cpu().reshape(-1)[0]

    if _is_int_dtype(dtype):
        np_t = _INT_NP[dtype]
        x1_np = x1.detach().cpu().numpy().astype(np_t)
        x2_np = x2.detach().cpu().numpy().astype(np_t)
        s_np = np.array(scalar.item(), dtype=np_t)
        # numpy 同 dtype 整型运算遵循 C 内建 2's-complement 回绕（与硬件一致），分两步截断。
        # 注意：np.asarray 保留 0-d（rank=0 标量）形状；不要用 ascontiguousarray（会把 0-d 提升为 [1]）。
        with np.errstate(over="ignore"):
            tmp = np.asarray(x1_np * s_np, dtype=np_t)      # 第一步：mul 回绕到 T
            y_np = np.asarray(tmp + x2_np, dtype=np_t)       # 第二步：add 回绕到 T
        # torch.from_numpy 不接受只读视图时复制一份；0-d 形状由 reshape(x1 形状) 保证一致
        return torch.from_numpy(y_np.copy()).reshape(x1.shape)

    # 浮点：在 float32 域计算 multiply -> add，再 cast 回目标 dtype
    x1_f = x1.detach().cpu().to(torch.float32)
    x2_f = x2.detach().cpu().to(torch.float32)
    scalar_f = scalar.to(torch.float32)
    y_f = x1_f * scalar_f + x2_f
    return y_f.to(dtype)


# ============================================================================
# Golden 正确性自测
# ============================================================================

def test_golden_correctness() -> bool:
    """验证 golden 函数本身的正确性（公式 / 标量广播 / 不变量 / 整型回绕 / 极端值）。"""
    all_passed = True

    def _check(name: str, got: torch.Tensor, expected: torch.Tensor, exact: bool = True,
               nan_mask: torch.Tensor = None) -> None:
        nonlocal all_passed
        if nan_mask is not None:
            # NaN 位置单独判定，其余位置数值判定
            got_nan = torch.isnan(got)
            ok = bool(torch.equal(got_nan, nan_mask))
            finite = ~nan_mask
            if exact:
                ok = ok and bool(torch.equal(got[finite], expected[finite]))
            else:
                ok = ok and bool(torch.allclose(got[finite].float(), expected[finite].float(),
                                                rtol=1e-3, atol=1e-3))
        elif exact:
            ok = bool(torch.equal(got, expected))
        else:
            ok = bool(torch.allclose(got.float(), expected.float(), rtol=1e-3, atol=1e-3))
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"    got     ={got.flatten().tolist()}")
            print(f"    expected={expected.flatten().tolist()}")
        all_passed = all_passed and ok

    # 1) fp32 基础公式 y = x1*x3[0] + x2
    x1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x2 = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    x3 = torch.tensor([2.0])
    _check("fp32 基础公式 y=x1*2+x2", compute_golden_fused_mul_add_n(x1, x2, x3),
           torch.tensor([[12.0, 24.0, 36.0], [48.0, 60.0, 72.0]]), exact=False)

    # 2) x3=0 不变量：y == x2
    x3z = torch.tensor([0.0])
    _check("x3=0 ⇒ y==x2 (zero_multiplier_yields_x2)",
           compute_golden_fused_mul_add_n(x1, x2, x3z), x2, exact=False)

    # 3) x3=1 退化为 addn：y == x1 + x2
    x3o = torch.tensor([1.0])
    _check("x3=1 ⇒ y==x1+x2 (单位元)",
           compute_golden_fused_mul_add_n(x1, x2, x3o), x1 + x2, exact=False)

    # 4) x3 形态 [1,1] 等价单元素标量广播
    _check("x3 形态[1,1] 等价标量",
           compute_golden_fused_mul_add_n(x1, x2, torch.tensor([[2.0]])),
           torch.tensor([[12.0, 24.0, 36.0], [48.0, 60.0, 72.0]]), exact=False)

    # 5) rank=0 标量输入
    _check("rank=0 标量输入",
           compute_golden_fused_mul_add_n(torch.tensor(3.0), torch.tensor(4.0), torch.tensor([2.0])),
           torch.tensor(10.0), exact=False)

    # 6) int32 基础（bitwise）
    ix1 = torch.tensor([10, 20, -30, 0, 50], dtype=torch.int32)
    ix2 = torch.tensor([5, -10, 30, 0, -50], dtype=torch.int32)
    ix3 = torch.tensor([3], dtype=torch.int32)
    _check("int32 基础 y=x1*3+x2",
           compute_golden_fused_mul_add_n(ix1, ix2, ix3),
           torch.tensor([35, 50, -60, 0, 100], dtype=torch.int32), exact=True)

    # 7) int32 上界回绕 INT32_MAX*1 + 1 = INT32_MIN（两步回绕）
    _check("int32 上界回绕 MAX*1+1=MIN",
           compute_golden_fused_mul_add_n(
               torch.tensor([2147483647], dtype=torch.int32),
               torch.tensor([1], dtype=torch.int32),
               torch.tensor([1], dtype=torch.int32)),
           torch.tensor([-2147483648], dtype=torch.int32), exact=True)

    # 8) int16 上界回绕 INT16_MAX*1 + 1 = INT16_MIN
    _check("int16 上界回绕 MAX*1+1=MIN",
           compute_golden_fused_mul_add_n(
               torch.tensor([32767], dtype=torch.int16),
               torch.tensor([1], dtype=torch.int16),
               torch.tensor([1], dtype=torch.int16)),
           torch.tensor([-32768], dtype=torch.int16), exact=True)

    # 9) int16 下界
    _check("int16 下界 MIN*1+0=MIN",
           compute_golden_fused_mul_add_n(
               torch.tensor([-32768, -32768], dtype=torch.int16),
               torch.tensor([0, 0], dtype=torch.int16),
               torch.tensor([1], dtype=torch.int16)),
           torch.tensor([-32768, -32768], dtype=torch.int16), exact=True)

    # 10) NaN 传播：x1 含 NaN ⇒ 输出该位置 NaN
    nan_x1 = torch.tensor([1.0, float("nan"), 3.0])
    nan_x2 = torch.tensor([1.0, 1.0, 1.0])
    g = compute_golden_fused_mul_add_n(nan_x1, nan_x2, torch.tensor([2.0]))
    _check("NaN 传播", g, torch.tensor([3.0, 0.0, 7.0]), exact=False,
           nan_mask=torch.tensor([False, True, False]))

    # 11) +inf：x1=+inf, x3=2, x2=1 ⇒ +inf
    inf_g = compute_golden_fused_mul_add_n(
        torch.tensor([float("inf")]), torch.tensor([1.0]), torch.tensor([2.0]))
    ok_inf = bool(torch.isinf(inf_g).all() and (inf_g > 0).all())
    print(f"  +inf 传播: {'PASS' if ok_inf else 'FAIL'}")
    all_passed = all_passed and ok_inf

    # 12) 全零输入 ⇒ y 全 0
    z = torch.zeros(8)
    _check("全零输入 ⇒ y 全 0",
           compute_golden_fused_mul_add_n(z, z, torch.zeros(1)), z, exact=False)

    return all_passed
