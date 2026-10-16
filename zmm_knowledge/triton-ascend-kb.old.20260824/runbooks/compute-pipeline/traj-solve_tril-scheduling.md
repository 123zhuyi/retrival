---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: solve_tril
dimension: "2.3"
title: Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行
description: 针对 Flash Linear Attention 中 (I+A)^(-1) 求解的调度策略，覆盖 BT=16/32/64 三层分级 kernel、16×16 原子递推、分治 merge 与 2D grid 并行，将 O(BT³) 求逆降为 O(BT²) 递推 + O(BT) 个 16×16 矩阵乘。
tags: [solve_tril, scheduling, hierarchical_dispatch, block_recursive, divide_and_conquer, grid_parallelism, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/solve_tril/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行

## 概述

`solve_tril` 计算 `(I + A)^(-1)`，其中 `A` 是严格下三角矩阵。若直接用稠密矩阵求逆，复杂度为 `O(BT³)`，且在 NPU 上无法利用三角结构的稀疏性。本卡唯一的 experience `exp_solve_tril_004` 采用**分层分治架构**：
1. 以 **16×16** 为原子计算单元，直接做逐行递推；
2. 对 **BT=32**，把 4 个 16×16 块递推后做一次 merge；
3. 对 **BT=64**，把 16 个 16×16 块递推后做 3 层 merge；
4. 用 **2D grid `(NT, B×H)`** 让每个 `(batch, head, chunk)` 三元组完全并行。

该架构将复杂度降到 `O(BT²)` 递推 + `O(BT)` 个 16×16 `tl.dot`，且是算子设计层面的决策，不是后期微优化。

**核心 trade-off**：
- 分治 merge 需要多个 kernel launch 和中间 workspace，但大幅降低了每个 kernel 的算法复杂度；
- 16×16 单元与 NPU Cube 粒度匹配，过小或过大都会损失效率；
- 三层分级增加了 host 调度复杂度，但换取了长序列（大 T）下的可扩展性。

## 共同模式

### 1. 三层分级 Dispatch：BT=16 / 32 / 64

**问题根因**：FLA 中 `BT`（chunk size）是算子语义参数，常见取值为 16、32、64。不同 BT 的最优递推/merge 模式不同，硬塞进一个 kernel 会导致大量分支和冗余循环。

**收益明显条件**：
- `BT ∈ {16, 32, 64}`；
- 需要支持多种 chunk size；
- 每个 chunk 内计算相互独立。

**实现要点**：
```python
if BT == 16:
    solve_tril_16x16_kernel[grid](...)
elif BT == 32:
    # 1) 递推 4 个对角 16×16 块
    solve_tril_16x16_kernel[grid](...)
    # 2) merge 得到非对角块
    merge_16x16_to_32x32_kernel[grid](...)
elif BT == 64:
    # 1) 递推 16 个对角 16×16 块
    solve_tril_16x16_kernel[grid](...)
    # 2) 三层 merge
    merge_16x16_to_32x32_kernel[grid](...)
    merge_32x32_to_64x64_kernel[grid](...)
```
- BT=16 直接求解；
- BT=32 用一次 merge；
- BT=64 用三层 merge（紧邻 → 隔行 → 最远）。

**常见坑**：
- 试图用运行时 `if BT == ...` 在单个 kernel 内处理，导致编译器无法静态展开；
- merge kernel 的输入/输出 workspace 未对齐，引发 bank conflict；
- BT=32 和 BT=64 的 merge 顺序错误会得到错误逆矩阵。

### 2. 16×16 原子递推单元

**问题根因**：Ascend NPU 的 Cube 单元对 16×16 矩阵乘支持最好。把 `BT` 拆成 16×16 块后，对角块可以直接用标量递推公式并行计算。

**收益明显条件**：
- `BT` 是 16 的倍数；
- 严格下三角结构已知；
- 对角块内计算量小但数量多。

**实现要点**：
```python
for i in range(1, 16):
    # 取 A 的第 i 行前 0..i-1 元素
    a_i = tl.extract_slice(neg_A, (i, 0), (1, 16), (1, 1))
    a_i = tl.reshape(a_i, (16,))
    # 递推：x_i = -a_i - sum_{k<i} a_i[k] * X[k]
    x_i = a_i + tl.sum(a_i[:, None] * X, axis=0)
    X = tl.where((tl.arange(0, 16) == i)[:, None], x_i, X)
```
- 每个 16×16 对角块内部逐行递推；
- 使用 `tl.extract_slice` 从预加载的寄存器块中取行；
- 用 `tl.where` 更新当前行，避免标量索引。

**常见坑**：
- 用动态索引 `A[i, :]` 导致 Triton 无法生成高效代码；
- 忘记对角线位置 `i==j` 应为 `1`（I+A 的对角线为 1）；
- 递推精度不够，未用 fp32 累加。

### 3. 分块递推公式：数学结构统一

**问题根因**：标量递推和分块递推在数学上同构。标量公式为：
```
X[i,j] = -A[i,j] - Σ_{k=j+1}^{i-1} A[i,k] * X[k,j]
```
分块公式为：
```
X[i,j] = -X[i,i] · Σ_{k=j}^{i-1} A[i,k] · X[k,j]
```
利用同构性，可以把 16×16 标量递推直接推广到 32×32/64×64 分块。

**收益明显条件**：
- 矩阵按 16×16 分块；
- 对角块已经求逆完成；
- 非对角块需要聚合多个子块贡献。

**实现要点**：
- 先计算所有对角块 `X[i,i]`；
- 再按列方向从左到右、行方向从上到下 merge；
- 每个 merge 步只做 16×16 `tl.dot`。

**常见坑**：
- 分块乘法顺序错误导致维度不匹配（应为 `(16,16) @ (16,16)`）；
- 忽略 `-X[i,i]` 前置负号；
- 非对角块的贡献漏加或多加。

### 4. 分治 Merge：X₂₁ = -X₂₂ · A₂₁ · X₁₁

**问题根因**：对于 32×32 块，把矩阵分成 4 个 16×16 子块：
```
[I+A] = | A11   0  |
        | A21  A22 |
X = [I+A]^-1 = | X11     0  |
               | X21   X22  |
```
其中 `X21 = -X22 · A21 · X11`。

**收益明显条件**：
- BT=32/64；
- 对角块已求逆；
- 需要计算非对角块。

**实现要点**：
```python
# X21 = -X22 @ A21 @ X11
X21 = -tl.dot(tl.dot(X22, A21), X11)
```
- 一次 merge 需要两次 16×16 `tl.dot`；
- BT=64 时，先 merge 出 32×32 块，再 merge 出 64×64 块；
- 每层 merge 的依赖方向要正确（只能依赖已经计算出的左/上块）。

**常见坑**：
- 矩阵块索引 `(i,j)` 与物理 offset 转换错误；
- 在 merge 中重复读取 A 块，应预加载到寄存器；
- 负数符号丢失导致结果错误。

### 5. Grid 并行：`(NT, B×H)` 二维调度

**问题根因**：所有 chunk 内的 `(batch, head)` 对相互独立。若用 1D grid，batch/head/chunk 的索引转换复杂；用 2D grid 可以让 `program_id(0)` 负责 chunk（`NT = T / BT`），`program_id(1)` 负责 batch/head。

**收益明显条件**：
- `B×H` 较大；
- `T` 是 `BT` 的倍数；
- 需要最大化 core 利用率。

**实现要点**：
```python
pid_t = tl.program_id(0)  # chunk id
pid_bh = tl.program_id(1) # batch * head id
batch_idx = pid_bh // H
head_idx = pid_bh % H
chunk_start = pid_t * BT
```
- 2D grid 天然把不同 batch/head 分配到不同 core；
- chunk 内所有计算在一个 program 内完成；
- 无跨 program 同步。

**常见坑**：
- 1D grid 硬编码上限 65535，大 T 或 B×H 大时容易溢出；
- `pid_bh` 到 batch/head 的除法未用 `tl.constexpr` 优化；
- 不同 BT 的 grid 计算方式不一致。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_solve_tril_004` | 算法复杂度优化 | 通过分治将 `O(BT³)` 稠密求逆降为 `O(BT²)` 递推 + `O(BT)` 个 16×16 矩阵乘；16×16 原子单元由 NPU 矩阵乘法硬件决定。 |

## 代表 Kernel

选择唯一的 `exp_solve_tril_004` 作为代表。下面展示 **BT=16 基础递推 kernel** 和 **BT=32 merge kernel** 的合成代码，体现三层分级调度中的核心模式：

```python
import triton
import triton.language as tl
import torch

@triton.jit
def solve_tril_16x16_kernel(
    A_ptr,
    X_ptr,
    stride_ab, stride_ah, stride_at,  # batch, head, tile
    stride_xb, stride_xh, stride_xt,
    B, H, NT,
    BT: tl.constexpr = 16,
):
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
    if pid_t >= NT or pid_bh >= B * H:
        return

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    # 当前 16×16 块偏移
    a_base = (
        A_ptr
        + batch_idx * stride_ab
        + head_idx * stride_ah
        + pid_t * BT * stride_at
    )
    x_base = (
        X_ptr
        + batch_idx * stride_xb
        + head_idx * stride_xh
        + pid_t * BT * stride_xt
    )

    # 加载 16×16 严格下三角 A（物理布局连续 BT×BT）
    offs = tl.arange(0, BT)
    A = tl.load(a_base + offs[:, None] * stride_at + offs[None, :])
    neg_A = -A.to(tl.float32)

    # X 初始化为单位矩阵
    X = tl.where(offs[:, None] == offs[None, :], 1.0, 0.0).to(tl.float32)

    # 逐行递推：X[i] = -A[i] - sum_{k<i} A[i,k] * X[k]
    for i in range(1, BT):
        a_i = tl.extract_slice(neg_A, (i, 0), (1, BT), (1, 1))
        a_i = tl.reshape(a_i, (BT,))
        x_i = a_i + tl.sum(a_i[:, None] * X, axis=0)
        X = tl.where((offs == i)[:, None], x_i, X)

    tl.store(x_base + offs[:, None] * stride_xt + offs[None, :], X)


@triton.jit
def merge_16x16_to_32x32_kernel(
    A_ptr,
    X_ptr,
    stride_ab, stride_ah, stride_at,
    stride_xb, stride_xh, stride_xt,
    B, H, NT,
    BT: tl.constexpr = 32,
):
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
    if pid_t >= NT or pid_bh >= B * H:
        return

    batch_idx = pid_bh // H
    head_idx = pid_bh % H
    a_base = A_ptr + batch_idx * stride_ab + head_idx * stride_ah + pid_t * BT * stride_at
    x_base = X_ptr + batch_idx * stride_xb + head_idx * stride_xh + pid_t * BT * stride_xt

    offs_16 = tl.arange(0, 16)

    # 分块索引：A = [[A11, 0], [A21, A22]]；X 同结构
    # 加载 X11, X22（已在前一步 kernel 求逆）
    X11 = tl.load(x_base + (offs_16[:, None] + 0) * stride_xt + (offs_16[None, :] + 0))
    X22 = tl.load(x_base + (offs_16[:, None] + 16) * stride_xt + (offs_16[None, :] + 16))
    A21 = tl.load(a_base + (offs_16[:, None] + 16) * stride_at + (offs_16[None, :] + 0))

    # X21 = -X22 @ A21 @ X11
    X21 = -tl.dot(tl.dot(X22, A21), X11)

    tl.store(x_base + (offs_16[:, None] + 16) * stride_xt + (offs_16[None, :] + 0), X21)
```

**关键实现细节**：
- 2D grid `(NT, B×H)`，每个 program 处理一个 chunk 的一个 batch/head；
- BT=16 基础 kernel 用 `tl.extract_slice` 做逐行递推，避免动态索引；
- BT=32 merge kernel 按分块公式 `X21 = -X22 @ A21 @ X11` 计算非对角块；
- BT=64 时，先调用 BT=16 基础 kernel，再依次调用 merge_16→32 和 merge_32→64；
- 所有累加使用 fp32，最后写回目标 dtype。

## 已知实例

- `exp_solve_tril_004` — Scheduling（收益 算法复杂度优化）
  - 通过分治将 `O(BT³)` 稠密求逆降为 `O(BT²)` 递推 + `O(BT)` 个 16×16 矩阵乘；16×16 原子单元由 NPU 矩阵乘法硬件决定。对应 kernel: `raw_experiences/extracted_kernels/153-solve_tril_16x16_kernel.py`、`154-merge_16x16_to_32x32_inverse_kernel.py`、`155-merge_16x16_to_64x64_inverse_kernel.py`。

## 关联知识

- 开发指南: [solve_tril 算子优化（三角矩阵求逆）](../../reference/op-class-solve_tril.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 主题相关：共享 910b2, 910b3
- 对应指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 主题相关：共享 910b2, 910b3
- 对应指南: [solve_tril 算子优化（三角矩阵求逆）](../../reference/op-class-solve_tril.md) — 主题相关：共享 910b2, 910b3, solve_tril
- 相关主题: [向量 gather 随机地址在 Ascend 上极慢，二分/搜索类走 per-expert 标量+grid 并行](../memory-sram/gather-random-address-slow.md) — 同属优化知识，主题相关
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](../memory-sram/traj-solve_tril-memory_access.md) — 主题相关：共享 910b2, 910b3, solve_tril
- 相关主题: [Ascend Triton 语法规范避坑聚合](../meta-skills/traj-general_ascend-syntax_spec.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Solve_Tril 优化决策：被排除的方向与根因](../meta-skills/traj-solve_tril-opt_decision.md) — 主题相关：共享 910b2, 910b3, solve_tril
- 相关主题: [Solve_Tril API 兼容性：从第三方仓库迁移时的参数审计与 TMA 不可用处理](../portability/traj-solve_tril-api_compat.md) — 主题相关：共享 910b2, 910b3, solve_tril
- 相关主题: [按输出元素数（而非输入）分配 grid，且钳制到 num_cores](grid-by-output-count.md) — 同属优化知识，主题相关
- 相关主题: [grid 总数不超过核数](grid-cap-to-core-num.md) — 同属优化知识，主题相关
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [grid_size 必须与 num_cores 入参严格相等](grid-strict-match-num-cores.md) — 同属优化知识，主题相关
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 同属优化知识，主题相关
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Attention Scheduling 优化聚合](traj-attention-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Embedding Backward Scheduling 优化聚合](traj-embedding-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Radix Sort Scatter 的 Lookback 调度优化](traj-gather_scatter-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [复杂算子拆分：降低编译与调度开销](traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Matmul Scheduling 优化聚合](traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Misc 算子 Scheduling 优化聚合](traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3, scheduling
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](traj-solve_tril-compute.md) — 主题相关：共享 910b2, 910b3, solve_tril

<!-- okf:related:end -->
