---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: solve_tril
dimension: "1.1"
title: Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读
description: 针对 solve_tril 循环内重复读取同一块 A 的内存访问瓶颈，覆盖 block_ptr 一次性预加载、tl.extract_slice 寄存器提取、动态索引限制与多 tile 扩展，BT=16/32/64 场景下 V4 相比 V1 加速 1.01x–1.37x。
tags: [solve_tril, memory_access, preload, extract_slice, register_reuse, block_ptr, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/solve_tril/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读

## 概述

`solve_tril` 的递推循环需要反复访问 `A` 的同一行/子块。最直观的写法是在循环内用 `tl.load(A + row_offset)` 逐行读取，但这会导致**同一块数据多次从 HBM 加载**。本卡唯一的 experience `exp_solve_tril_002` 采用 **Preload + `tl.extract_slice`** 策略：
1. 循环外通过 `tl.make_block_ptr` 一次性把整个 `BT×BT` 块加载到寄存器；
2. 循环内用 `tl.extract_slice` 从寄存器中提取当前需要的行或子块；
3. 避免 Triton 不支持的动态索引 `b_A[i, :]`。

实测效果（V4 vs V1）：
- BT=16：1.01x–1.37x；
- BT=32：1.25x–1.27x（首次优化）；
- BT=64：1.11x–1.13x。

与 `exp_solve_tril_001`（冗余 mask 消除）组合后，ORIGINAL → V4 累计加速比：BT=16 达 1.42x–2.54x，BT=32 达 1.26x–1.29x，BT=64 达 1.55x–1.57x。

**核心 trade-off**：
- 预加载增加寄存器占用，但能消除循环内 HBM 重读；
- `extract_slice` 偏移必须是编译器可表达的 slice，不能是任意 gather；
- 小 grid 下收益会被额外寄存器操作抵消。

## 共同模式

### 1. 循环外一次性预加载整个块

**问题根因**：如果循环内每次迭代都 `tl.load(A + i * stride)`，即使 `A` 的同一行/列已经在前一次迭代附近，每次 load 仍然是独立的 HBM 访问，无法被 L0/L1/L2 完全命中。

**收益明显条件**：
- 循环次数 ≥ 2；
- 每次迭代访问的数据都在同一个 `BT×BT` 块内；
- `BT×BT` 块大小不超过寄存器/UB 容量。

**实现要点**：
```python
# 循环外：用 block_ptr 加载整个 BT×BT 块
b_A = tl.load(tl.make_block_ptr(
    base=A_ptr + tile_offset,
    shape=(BT, BT),
    strides=(stride_at, 1),
    offsets=(0, 0),
    block_shape=(BT, BT),
    order=(1, 0),
))
neg_A = -b_A.to(tl.float32)
```
- 一次性加载 `BT×BT`；
- 预先把符号取反，避免循环内重复计算；
- 使用 `boundary_check` 处理尾部 tile。

**常见坑**：
- 只预加载了当前行，没有预加载整个块；
- 预加载后又在循环内重复 `tl.load`；
- 预加载的块太大导致寄存器 spill。

### 2. 循环内用 `tl.extract_slice` 提取子块

**问题根因**：Triton 的 block tensor 不支持运行时动态索引（`b_A[i, :]` 会报错）。`tl.extract_slice` 是官方提供的寄存器子集提取原语，offset 可以是运行时变量，但只支持连续 slice。

**收益明显条件**：
- 数据已经在寄存器中；
- 需要按行、列或固定子块访问；
- 子块是连续的（stride 固定）。

**实现要点**：
```python
for i in range(2, BT):
    # 从预加载的 neg_A 中提取第 i 行
    b_a = tl.extract_slice(neg_A, (i, 0), (1, BT), (1, 1))
    b_a = tl.reshape(b_a, (BT,))
    # 递推计算
    x_i = b_a + tl.sum(b_a[:, None] * X, axis=0)
    X = tl.where((tl.arange(0, BT) == i)[:, None], x_i, X)
```
- `extract_slice` 的 offsets/sizes/strides 必须合法；
- 提取后通常需要 `reshape` 到合适维度；
- 仍在寄存器内，不触发 HBM 访问。

**常见坑**：
- 用 `b_A[i, :]` 导致编译失败；
- `extract_slice` 越界（offset+size 超过预加载块大小）；
- 提取后 reshape 维度错误。

### 3. 为什么不能用动态索引

**问题根因**：Triton 的 tensor 索引在编译期需要确定 shape 和 stride 模式。运行时的 int32 索引会破坏这种静态分析，编译器无法生成高效的 block transfer 或 broadcast。

**错误示例**：
```python
# 编译报错：unsupported tensor index: int32[]
b_a = b_A[i, :]
```

**替代方案**：
- 如果访问模式是固定子块：用 `tl.extract_slice`；
- 如果访问模式是编译期已知：用 `tl.constexpr` 索引或展开循环；
- 如果必须动态 gather：退化为 `tl.load` 是最后手段，但要接受性能损失。

**常见坑**：
- 把 Python `for i in range(BT)` 中的 `i` 当成编译期常量——它实际上是运行时 int32；
- 用 `tl.arange` 和 mask 组合模拟索引，但生成冗余 ALU；
- 在循环内用 `tl.load(ptr + i * stride)` 而没有预加载。

### 4. 多 tile 扩展：BT=32/64

**问题根因**：BT=32/64 时，solve_tril 会先对 16×16 对角块做递推，再 merge 非对角块。每个 16×16 对角块都可以独立预加载自己的 `A` 子块。

**收益明显条件**：
- `BT` 是 16 的倍数；
- 采用分层分治架构（见 `traj-solve_tril-scheduling.md`）；
- 每个对角 tile 的 `A` 子块在循环内被多次访问。

**实现要点**：
```python
# 对 BT=32，加载两个 16×16 对角 A 子块
for diag in range(BT // 16):
    a_sub = tl.load(tl.make_block_ptr(
        base=A_ptr + tile_offset + diag * 16 * stride_at + diag * 16,
        shape=(16, 16),
        strides=(stride_at, 1),
        offsets=(0, 0),
        block_shape=(16, 16),
        order=(1, 0),
    ))
    neg_A_sub = -a_sub.to(tl.float32)
    for i in range(1, 16):
        b_a = tl.extract_slice(neg_A_sub, (i, 0), (1, 16), (1, 1))
        ...
```
- 每个 16×16 对角块独立预加载；
- merge 阶段同样用 `extract_slice` 或 `tl.dot` 从寄存器读取已加载的 A 子块。

**常见坑**：
- 只预加载了当前对角块，但 merge 阶段又重复读取 A 的非对角块；
- 多 tile 时 workspace 布局与预加载块的 offset 不对应；
- 预加载块之间重叠导致数据重复。

### 5. 与 constexpr 双路径结合

**问题根因**：绝大多数 chunk 是完整 tile（`T % BT == 0`），最后一个 chunk 可能是边界 tile。预加载 + extract_slice 可以和 `if T - offset >= BT` 的 constexpr 双路径结合：完整 tile 走完全展开的快速路径，边界 tile 仍从预加载块提取但用运行时循环。

**收益明显条件**：
- `T` 很大，99% 以上为完整 tile；
- 需要同时处理边界 tile；
- 希望完整 tile 获得 constexpr 展开收益。

**实现要点**：
```python
b_A = tl.load(...)  # 预加载，对完整 tile 是 BT×BT，对边界 tile 是 remaining×BT
if T - offset >= BT:
    # 快速路径：constexpr 展开
    for i in tl.static_range(2, BT):
        b_a = tl.extract_slice(neg_A, (i, 0), (1, BT), (1, 1))
        ...
else:
    # 慢速路径：运行时循环
    for i in range(2, min(BT, T - offset)):
        b_a = tl.extract_slice(neg_A, (i, 0), (1, BT), (1, 1))
        ...
```
- 完整 tile 用 `tl.static_range` 编译期展开；
- 边界 tile 用运行时 `range`，但仍从寄存器提取。

**常见坑**：
- 把 `b_A` 的 shape 在边界 tile 设为 `(BT, BT)`，导致加载越界；
- 边界 tile 的 `extract_slice` 仍用 `(1, BT)`，应改为 `(1, remaining)`；
- 两个路径的 mask/边界处理不一致。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_solve_tril_002` | BT=16：1.01x–1.37x；BT=32：1.25x–1.27x（首次优化）；BT=64：1.11x–1.13x | V4 核心优化：循环外通过 block_ptr 预加载整个 BT×BT 块，循环内用 `tl.extract_slice` 从寄存器提取行/子块，避免重复 HBM 读取。与 V1 组合后累计加速比：BT=16 达 1.42x–2.54x，BT=32 达 1.26x–1.29x，BT=64 达 1.55x–1.57x。 |

## 代表 Kernel

选择唯一的 `exp_solve_tril_002` 作为代表。下面为合成的 BT=16 solve_tril kernel，展示 **block_ptr 预加载 + `tl.extract_slice` 寄存器提取** 的核心内存访问模式：

```python
import triton
import triton.language as tl
import torch

@triton.jit
def solve_tril_preload_kernel(
    A_ptr,
    X_ptr,
    stride_ab, stride_ah, stride_at,
    stride_xb, stride_xh, stride_xt,
    B, H, NT, T,
    BT: tl.constexpr = 16,
):
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)
    if pid_t >= NT or pid_bh >= B * H:
        return

    batch_idx = pid_bh // H
    head_idx = pid_bh % H
    offset = pid_t * BT
    remaining = T - offset

    a_base = (
        A_ptr
        + batch_idx * stride_ab
        + head_idx * stride_ah
        + offset * stride_at
    )
    x_base = (
        X_ptr
        + batch_idx * stride_xb
        + head_idx * stride_xh
        + offset * stride_xt
    )

    # ---------- 1. 循环外一次性预加载 BT×BT 块 ----------
    # 边界 tile 时实际 shape 为 (remaining, BT)，block_shape 仍为 (BT, BT)
    a_block_ptr = tl.make_block_ptr(
        base=a_base,
        shape=(remaining, BT),
        strides=(stride_at, 1),
        offsets=(0, 0),
        block_shape=(BT, BT),
        order=(1, 0),
    )
    b_A = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
    neg_A = -b_A.to(tl.float32)

    # X 初始化为单位矩阵
    offs = tl.arange(0, BT)
    X = tl.where(offs[:, None] == offs[None, :], 1.0, 0.0).to(tl.float32)

    # ---------- 2. 循环内用 extract_slice 从寄存器提取 ----------
    if remaining >= BT:
        # 完整 tile：constexpr 展开
        for i in tl.static_range(1, BT):
            b_a = tl.extract_slice(neg_A, (i, 0), (1, BT), (1, 1))
            b_a = tl.reshape(b_a, (BT,))
            x_i = b_a + tl.sum(b_a[:, None] * X, axis=0)
            X = tl.where((offs == i)[:, None], x_i, X)
    else:
        # 边界 tile：运行时循环，但仍从寄存器提取
        for i in range(1, remaining):
            b_a = tl.extract_slice(neg_A, (i, 0), (1, BT), (1, 1))
            b_a = tl.reshape(b_a, (BT,))
            x_i = b_a + tl.sum(b_a[:, None] * X, axis=0)
            X = tl.where((offs == i)[:, None], x_i, X)

    tl.store(
        x_base + offs[:, None] * stride_xt + offs[None, :],
        X,
        mask=(offs[:, None] < remaining) & (offs[None, :] < BT),
    )
```

**关键实现细节**：
- 循环外通过 `tl.make_block_ptr` 一次性加载整个 `BT×BT`（或边界 tile 的 `remaining×BT`）；
- 循环内用 `tl.extract_slice(neg_A, (i, 0), (1, BT), (1, 1))` 从寄存器提取第 `i` 行，不访问 HBM；
- 完整 tile 走 `tl.static_range` 编译期展开；
- 边界 tile 走运行时循环，但同样使用预加载数据；
- store 时用 mask 保护边界，避免写入 padding 区。

## 已知实例

- `exp_solve_tril_002` — Memory Access（收益 BT=16：1.01x–1.37x；BT=32：1.25x–1.27x；BT=64：1.11x–1.13x）
  - V4 核心优化：循环外预加载整个 BT×BT 块，循环内用 `tl.extract_slice` 从寄存器提取，避免重复 HBM 读取。与 V1 组合后累计加速比：BT=16 达 1.42x–2.54x，BT=32 达 1.26x–1.29x，BT=64 达 1.55x–1.57x。

## 关联知识

- 开发指南: [solve_tril 算子优化（三角矩阵求逆）](../../reference/op-class-solve_tril.md)
- 优化原则: [preload + extract_slice 消除循环内同一数据重复 HBM 读取](preload-extract-slice-eliminate-reread.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 主题相关：共享 910b2, 910b3
- 对应指南: [solve_tril 算子优化（三角矩阵求逆）](../../reference/op-class-solve_tril.md) — 主题相关：共享 910b2, 910b3, solve_tril
- 相关主题: [FlashAttention 变体的内存访问优化](../algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](../compute-pipeline/traj-solve_tril-compute.md) — 主题相关：共享 910b2, 910b3, extract_slice, solve_tril
- 相关主题: [Solve_Tril 调度优化：三层分级 kernel、分块递推与 Grid 并行](../compute-pipeline/traj-solve_tril-scheduling.md) — 主题相关：共享 910b2, 910b3, solve_tril
- 相关主题: [Solve_Tril 优化决策：被排除的方向与根因](../meta-skills/traj-solve_tril-opt_decision.md) — 主题相关：共享 910b2, 910b3, solve_tril
- 相关主题: [Ascend 910B 硬件约束：i64 地址、整数比较与 block_ptr boundary_check 标量降级](../portability/traj-general_ascend-hardware_constraint.md) — 主题相关：共享 910b2, 910b3, block_ptr
- 相关主题: [Solve_Tril API 兼容性：从第三方仓库迁移时的参数审计与 TMA 不可用处理](../portability/traj-solve_tril-api_compat.md) — 主题相关：共享 910b2, 910b3, solve_tril
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](block-ptr-vs-scalar-load.md) — 主题相关：共享 block_ptr
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 主题相关：共享 memory_access
- 相关主题: [preload + extract_slice 消除循环内同一数据重复 HBM 读取](preload-extract-slice-eliminate-reread.md) — 主题相关：共享 block_ptr, extract_slice, preload, register_reuse
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](register-shift-sliding-window.md) — 同属优化知识，主题相关
- 相关主题: [Activation Backward 内存访问优化聚合](traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Gather/Scatter 内存访问优化聚合](traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, block_ptr, memory_access, register_reuse
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Misc 算子内存访问优化聚合](traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Sum-and-Scale 归约内存访问优化](traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Softmax Backward 内存访问优化聚合](traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, register_reuse

<!-- okf:related:end -->
