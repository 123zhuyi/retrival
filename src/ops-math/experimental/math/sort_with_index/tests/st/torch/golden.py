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
CPU Golden 计算模块 —— SortWithIndex（experimental/math/sort_with_index，ascend910b）

================================================================================
⚠️ 910B 实现语义：
   本 golden 严格对齐 C++ ST tests/st/test_aclnn_sort_with_index.cpp::ComputeGolden910b，
   **不可** 使用 torch.sort 的默认 NaN 落位约定（torch 升序 NaN 落末尾）。

   910B 升序 = Muls(-1) + 硬件降序 Sort + Muls(-1)（不反转名次）：
     - 升序（descending=false）：NaN 落序列「开头」（rank0 起；偏离 torch 升序的末尾约定）。
     - 降序（descending=true）： NaN 落序列「开头」（与 torch 约定一致）。
   故无论升降序，**NaN 都落开头**。

   有限值/±Inf 区间：+Inf > 有限 > -Inf，按数值大小正常排序；±0 视为相等（ties）。
   stable：std::stable_sort 等价 —— NaN 之间 / ties 之间保持原始位置升序。

   NaN 位型：升序 Muls(-1) 翻 NaN 符号位（输出 bit ≠ 输入 canonical），值仍是 NaN；
   value 比对对 NaN 行按 isnan，不按 bit（见 compare.py）。

   这是「索引跟随排序」重排算子：y[k]=x[p[k]]、sorted_index[k]=index[p[k]]，同一 permutation p。
   torch.sort 只重排自身 index，不支持外部传入 index 张量跟随重排，故无现成 oracle，
   替代 golden = 本模块（与 C++ std::stable_sort 910B 语义对齐）。
================================================================================
"""

import numpy as np
import torch


# ============================================================================
# 排序键投影（910B kernel：bf16/int32 经 float 排序）
#   - float16 / float32 / bfloat16 → 投影为 float（序关系不变；bf16→float 精确）
#   - int32 value                  → 投影为 float（限 |x|<=2^24 精确）
# ============================================================================

def _project_key_np(x_np: np.ndarray) -> np.ndarray:
    """把任意 value dtype 投影为 float64 排序键（与 910B 经 float 排序的序关系一致）。"""
    return x_np.astype(np.float64)


def _sort_one_row_perm(keys: np.ndarray, descending: bool) -> np.ndarray:
    """
    对一行排序键 keys 求 910B 语义置换 p（NaN 落开头、stable）。

    返回 p：长度 N 的 int64 索引数组，y[k]=x[p[k]]。

    910B 比较语义（与 C++ ComputeGolden910b 一致）：
      1) NaN 元素稳定放序列开头（保持原始位置升序）；
      2) 其余有限值/±Inf 按 ascending(descending=False) / descending(=True) 稳定排序。

    实现：构造两路稳定排序键。
      - primary：NaN → 0（排最前），非 NaN → 1（排其后）。
      - secondary（仅非 NaN 内有效）：升序用 keys，降序用 -keys。
      np.lexsort 是稳定排序（ties 保持原始下标升序），等价 std::stable_sort。
    """
    n = keys.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int64)

    is_nan = np.isnan(keys)
    primary = np.where(is_nan, 0, 1).astype(np.int64)  # NaN 在前

    # secondary：非 NaN 按 ascending/descending；NaN 行 secondary 置 0（primary 已把 NaN 拉到最前，
    # NaN 之间 stable 由 lexsort 的稳定性 + 原始下标升序保证）。
    secondary = np.zeros(n, dtype=np.float64)
    finite_mask = ~is_nan
    if descending:
        secondary[finite_mask] = -keys[finite_mask]
    else:
        secondary[finite_mask] = keys[finite_mask]

    # np.lexsort：最后一个 key 是主键。主键 primary（NaN 在前），次键 secondary。
    # lexsort 稳定 → 同 (primary, secondary) 的 ties 按原始下标升序（= std::stable_sort 语义）。
    perm = np.lexsort((secondary, primary))
    return perm.astype(np.int64)


def compute_golden_sort_with_index(
    x: torch.Tensor,
    index: torch.Tensor,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
):
    """
    SortWithIndex CPU golden（910B 语义）。

    Args:
        x:          待排序数值张量（float16/float32/bfloat16/int32）
        index:      待跟随排序的索引张量（int32），shape 与 x 一致
        axis:       排序轴（当前仅支持最后一维，-1 或 rank-1）
        descending: True 降序、False 升序
        stable:     语义占位（golden 一律 stable，tie-break 原始位置升序；
                    stable=false 时 NPU 仍多为稳定，结果无害）

    Returns:
        (y, sorted_index): torch.Tensor，dtype/shape 与 (x, index) 一致。
          y[k] = x[p[k]]，sorted_index[k] = index[p[k]]（同一 permutation p）。

    NaN 升序/降序均落开头；NaN 位型按 910B Muls(-1) 还原后仍是 NaN（比对按 isnan）。
    """
    _ = stable  # golden 一律稳定（tie-break 原始位置升序）
    if x.shape != index.shape:
        raise ValueError(f"x.shape {tuple(x.shape)} != index.shape {tuple(index.shape)}")

    rank = x.dim()
    # axis 规范化到最后一维（910B 仅支持最后一维）。
    norm_axis = axis if axis >= 0 else axis + rank
    if rank > 0 and norm_axis != rank - 1:
        raise ValueError(f"only last-axis sort supported; axis={axis} (rank={rank})")

    # rank0 标量：单元素直接拷贝。
    if rank == 0:
        return x.clone(), index.clone()

    # bf16 需先 view 成 float 做 numpy 计算（numpy 无 bfloat16）；保留原 dtype 还原 y。
    x_is_bf16 = (x.dtype == torch.bfloat16)
    x_cpu = x.detach().cpu()
    idx_cpu = index.detach().cpu()

    if x_is_bf16:
        # bf16 -> float32（精确，序关系不变），numpy 计算后还原 bf16。
        x_key_t = x_cpu.to(torch.float32)
        x_np_key = x_key_t.numpy()
        # value 存储：用 uint16 位型跟随置换（保证 bitwise 搬移，无量化误差）。
        x_store = x_cpu.view(torch.int16).numpy()  # bf16 位型（int16 容器）
    else:
        x_np_key = x_cpu.numpy().astype(np.float64) if x_cpu.dtype != torch.float64 else x_cpu.numpy()
        x_store = x_cpu.numpy()

    idx_np = idx_cpu.numpy()

    axis_len = x_cpu.shape[-1]
    flat_shape = (-1, axis_len) if axis_len > 0 else (0, 0)

    key_2d = x_np_key.reshape(-1, axis_len) if axis_len > 0 else x_np_key.reshape(0, 0)
    store_2d = x_store.reshape(-1, axis_len) if axis_len > 0 else x_store.reshape(0, 0)
    idx_2d = idx_np.reshape(-1, axis_len) if axis_len > 0 else idx_np.reshape(0, 0)

    y_store_2d = np.empty_like(store_2d)
    si_2d = np.empty_like(idx_2d)

    rows = key_2d.shape[0]
    for r in range(rows):
        if axis_len == 0:
            continue
        perm = _sort_one_row_perm(_project_key_np(key_2d[r]), descending)
        y_store_2d[r] = store_2d[r][perm]
        si_2d[r] = idx_2d[r][perm]

    # 还原 dtype / shape。
    if x_is_bf16:
        y_store = y_store_2d.reshape(x_cpu.shape)
        y = torch.from_numpy(y_store).view(torch.bfloat16)
    else:
        y_store = y_store_2d.reshape(x_cpu.shape)
        y = torch.from_numpy(y_store).to(x_cpu.dtype)

    sorted_index = torch.from_numpy(si_2d.reshape(idx_cpu.shape)).to(idx_cpu.dtype)
    return y, sorted_index


# ============================================================================
# Golden 正确性自测（910B NaN-开头 语义；对齐 C++ ST TestGoldenCorrectness 自测 1–9）
# ============================================================================

def _isnan_t(t: torch.Tensor) -> torch.Tensor:
    """对任意 dtype 安全求 isnan（整型恒 False）。"""
    if t.dtype.is_floating_point or t.dtype == torch.bfloat16:
        return torch.isnan(t.float())
    return torch.zeros_like(t, dtype=torch.bool)


def test_golden_correctness() -> bool:
    """验证 golden 函数本身的正确性（910B NaN-开头）。逐项对齐 C++ ST 自测。"""
    all_passed = True

    def check(name, cond):
        nonlocal all_passed
        print(f"  {name}: {'PASS' if cond else 'FAIL'}")
        all_passed = all_passed and bool(cond)

    nan = float("nan")
    pinf = float("inf")
    ninf = float("-inf")

    # 自测 1：升序 + ties stable（shape=[5]，fp16）
    x = torch.tensor([3.0, 1.0, 4.0, 1.0, 2.0], dtype=torch.float16)
    idx = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    y, si = compute_golden_sort_with_index(x, idx, -1, False, True)
    # 1,1,2,3,4 -> idx 1,3,4,0,2
    check("自测1 升序+ties stable[5]",
          torch.equal(si, torch.tensor([1, 3, 4, 0, 2], dtype=torch.int32))
          and torch.equal(y.float(), torch.tensor([1.0, 1.0, 2.0, 3.0, 4.0])))

    # 自测 2：降序（shape=[5]）
    y, si = compute_golden_sort_with_index(x, idx, -1, True, True)
    # 4,3,2,1,1 -> idx 2,0,4,1,3
    check("自测2 降序+ties stable[5]",
          torch.equal(si, torch.tensor([2, 0, 4, 1, 3], dtype=torch.int32))
          and torch.equal(y.float(), torch.tensor([4.0, 3.0, 2.0, 1.0, 1.0])))

    # 自测 3：多行独立排序（shape=[2,3]）
    x3 = torch.tensor([[3.0, 1.0, 2.0], [9.0, 7.0, 8.0]], dtype=torch.float16)
    idx3 = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32)
    y, si = compute_golden_sort_with_index(x3, idx3, -1, False, True)
    check("自测3 多行独立排序[2,3]",
          torch.equal(si, torch.tensor([[1, 2, 0], [1, 2, 0]], dtype=torch.int32)))

    # 自测 4：★ 910B NaN 升序落「开头」（非 torch 末尾）
    x4 = torch.tensor([2.0, nan, 1.0, 3.0], dtype=torch.float16)
    idx4 = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    y, si = compute_golden_sort_with_index(x4, idx4, -1, False, True)
    # 910B 升序: NaN(开头),1,2,3 -> idx 1,2,0,3
    check("自测4 ★NaN升序落开头[4] idx=1,2,0,3",
          torch.equal(si, torch.tensor([1, 2, 0, 3], dtype=torch.int32))
          and bool(_isnan_t(y)[0]))

    # 自测 5：降序 NaN 也落开头
    y, si = compute_golden_sort_with_index(x4, idx4, -1, True, True)
    # 910B 降序: NaN(开头),3,2,1 -> idx 1,3,0,2
    check("自测5 降序NaN落开头[4] idx=1,3,0,2",
          torch.equal(si, torch.tensor([1, 3, 0, 2], dtype=torch.int32))
          and bool(_isnan_t(y)[0]))

    # 自测 6：±Inf 落位（升序 -Inf 开头、+Inf 末尾）
    x6 = torch.tensor([2.0, pinf, ninf, 1.0], dtype=torch.float16)
    idx6 = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    y, si = compute_golden_sort_with_index(x6, idx6, -1, False, True)
    # 升序: -Inf,1,2,+Inf -> idx 2,3,0,1
    check("自测6 ±Inf升序(-Inf开头/+Inf末尾)[4]",
          torch.equal(si, torch.tensor([2, 3, 0, 1], dtype=torch.int32)))

    # 自测 7：NaN+±Inf 综合（升序：NaN(开头), -Inf, 有限, +Inf）
    x7 = torch.tensor([ninf, 5.0, nan, pinf], dtype=torch.float32)
    idx7 = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    y, si = compute_golden_sort_with_index(x7, idx7, -1, False, True)
    # 升序: NaN(开头),-Inf,5,+Inf -> idx 2,0,1,3
    check("自测7 NaN+±Inf综合升序[4] idx=2,0,1,3",
          torch.equal(si, torch.tensor([2, 0, 1, 3], dtype=torch.int32))
          and bool(_isnan_t(y)[0]))

    # 自测 8：空 tensor（shape=[0]）
    xe = torch.empty((0,), dtype=torch.float16)
    ie = torch.empty((0,), dtype=torch.int32)
    y, si = compute_golden_sort_with_index(xe, ie, -1, False, True)
    check("自测8 空tensor[0] 返回空", y.numel() == 0 and si.numel() == 0)

    # 自测 9：int32 value golden（Cast 语义，|x|<=2^24）
    xi = torch.tensor([30, -10, 20, -10, 5], dtype=torch.int32)
    idxi = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    y, si = compute_golden_sort_with_index(xi, idxi, -1, False, True)
    # 升序 -10,-10,5,20,30 -> idx 1,3,4,2,0
    check("自测9 int32 value 升序+ties[5]",
          torch.equal(y, torch.tensor([-10, -10, 5, 20, 30], dtype=torch.int32))
          and torch.equal(si, torch.tensor([1, 3, 4, 2, 0], dtype=torch.int32)))

    # 自测 10：bf16 value（位型跟随搬移，序关系经 float 一致）
    xb = torch.tensor([3.0, 1.0, 2.0], dtype=torch.bfloat16)
    idxb = torch.tensor([0, 1, 2], dtype=torch.int32)
    y, si = compute_golden_sort_with_index(xb, idxb, -1, False, True)
    check("自测10 bf16 升序[3]",
          y.dtype == torch.bfloat16
          and torch.equal(si, torch.tensor([1, 2, 0], dtype=torch.int32))
          and torch.equal(y.float(), torch.tensor([1.0, 2.0, 3.0])))

    # 自测 11：rank0 标量直接拷贝
    xs = torch.tensor(42.0, dtype=torch.float32)
    iss = torch.tensor(7, dtype=torch.int32)
    y, si = compute_golden_sort_with_index(xs, iss, -1, False, True)
    check("自测11 rank0 标量拷贝", torch.equal(y, xs) and torch.equal(si, iss))

    # 自测 12：±0 视为相等 ties（stable index 升序）
    xz = torch.tensor([0.0, -0.0, 0.0, -0.0], dtype=torch.float32)
    idxz = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    y, si = compute_golden_sort_with_index(xz, idxz, -1, False, True)
    # 全相等 ties -> stable 原始位置升序 0,1,2,3
    check("自测12 ±0 ties stable index 升序",
          torch.equal(si, torch.tensor([0, 1, 2, 3], dtype=torch.int32)))

    print(f"\nGolden 自测汇总: {'全部 PASS' if all_passed else '存在 FAIL'}")
    return all_passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if test_golden_correctness() else 1)
