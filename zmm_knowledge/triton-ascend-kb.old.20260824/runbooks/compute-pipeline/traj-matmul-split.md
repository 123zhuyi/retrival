---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: matmul
dimension: "2.3"
title: Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略
description: 聚合 3 条 matmul split 经验，覆盖 A/B 不转置、B 转置、A/B 双转置三种场景下的分块切分策略，核心约束为 Ascend Cube-Vector 行宽 512B 对齐，收益 10%–32%。
tags: [matmul, split, tiling, 512b_alignment, block_pointer, transpose, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/matmul/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略

## 概述

Matmul 在 Ascend 910B 上的性能很大程度上由**分块尺寸（BLOCK_M/BLOCK_K/BLOCK_N）**决定。`matmul Split` 这组经验专门讨论：在不同转置组合（A/B 是否转置）下，如何选择分块形状以满足：
1. **Cube-Vector 行宽 512B 对齐**：`BLOCK_N * dtype_bytes` 必须是 512 的整数倍，否则触发 partial cache-line read；
2. **数据复用最大化**：让 K 维切片在 L0A/L0B 中复用，中间累加器留在 L0C；
3. **转置后的访问连续**：转置会改变内存中的连续维度，分块需适配实际 stride。

本卡 3 条 experience 分别覆盖：
- `exp_matmul_003`（10%）：A、B 均不转置；
- `exp_matmul_004`（10%）：A 不转置、B 转置；
- `exp_matmul_005`（32%）：A、B 均转置，收益最高。

**核心 trade-off**：
- 更大的 BLOCK_K 提升 K 维复用，但增加 L0A/L0B 占用；
- BLOCK_N 必须满足 512B 对齐，可选项受限；
- 转置场景下不能照搬不转置的分块，需要按物理 stride 调整 BLOCK_N/BLOCK_K。

## 共同模式

### 1. 512B 行宽对齐：BLOCK_N/BLOCK_K 的硬性约束

**问题根因**：Ascend Cube 单元做向量化 load 时，若 tile 行宽不是 512B 整数倍，会触发 partial cache-line read，MTE2 带宽骤降，且 bank conflict 概率增加。

**收益明显条件**：
- 任何使用 `tl.dot` 或 Cube-Vector 向量化 load 的 matmul；
- dtype 为 fp16/bf16（512B 对应 256 个元素）或 fp32（对应 128 个元素）；
- 大矩阵场景下不对齐的损失更明显。

**实现要点**：
```python
# fp16/bf16：BLOCK_N、BLOCK_K 至少为 256 的倍数（或 128 若满足整体 512B）
# fp32：BLOCK_N、BLOCK_K 至少为 128 的倍数
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE': 16}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 256, 'GROUP_SIZE': 16}),
    ],
    key=['M', 'N', 'K'],
)
```
- 优先固定 `BLOCK_K=256`；
- `BLOCK_N` 在 fp16/bf16 下取 128 或 256；
- autotune 候选必须满足对齐约束。

**常见坑**：
- 把 `BLOCK_N=100` 这类非对齐值加入搜索空间；
- 只关注 M/N 维度，忽略 K 维切片也需对齐；
- 转置后误以为逻辑 `BLOCK_N` 就是物理行宽。

### 2. `tl.make_block_ptr` 保持合并访存

**问题根因**：手动计算 `ptr + i * stride` 进行多次标量 `tl.load` 会退化为离散地址访问，丧失 block transfer 的 burst 合并与预取能力。

**收益明显条件**：
- 所有 2D tile 级 load；
- 分块尺寸固定且已知；
- 需要 `boundary_check` 处理尾部 tile。

**实现要点**：
```python
a_block_ptr = tl.make_block_ptr(
    base=a_ptr,
    shape=(M, K),
    strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, 0),
    block_shape=(BLOCK_M, BLOCK_K),
    order=(1, 0),
)
for _ in range(0, K, BLOCK_K):
    a = tl.load(a_block_ptr, boundary_check=(0, 1))
    ...
    a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
```
- 用 `make_block_ptr` 描述整个 tile 的 shape/stride/offset；
- K 循环中用 `tl.advance` 推进；
- 转置场景下 `shape`/`strides`/`order` 要对应物理张量。

**常见坑**：
- 在 K 循环内重新构造 block_ptr，增加地址计算开销；
- `order` 与物理 layout 不一致导致 stride 计算错误；
- 转置时 `shape` 仍用逻辑 shape 而非物理 shape。

### 3. 不转置场景：标准列优先 PID 分组

**问题根因**：A、B 均不转置时，B 的物理 layout 是 (K, N)。K 循环中每次加载 B 的同一行不同列，相邻 program 若共享 K 块，可提升 L2 命中率。

**收益明显条件**：
- `trans_a=False, trans_b=False`；
- M、N 均较大（≥256）；
- 需要提升 L2 局部性。

**实现要点**：
- 使用列优先 PID 分组（`GROUP_SIZE` 控制同组 program 共享 A 的行块）；
- `BLOCK_M` 和 `BLOCK_N` 均取 128/256；
- `BLOCK_K=256` 固定。

**常见坑**：
- `GROUP_SIZE` 过大导致同组 program 过多，L2 竞争；
- 忽略 B 的 K 维连续，导致每次 K 迭代 load 不连续；
- BLOCK_M 过小无法填满 Cube。

### 4. B 转置场景：K 宽统一分块

**问题根因**：`trans_b=True` 时，B 的物理 shape 是 (N, K)。K 循环中访问 B 的列方向，为了连续访问，需要让 `BLOCK_K` 成为 512B 对齐的固定宽度，且 N 维分块可按行读取。

**收益明显条件**：
- `trans_a=False, trans_b=True`；
- K 较大（≥256）；
- B 的物理 layout 为 (N,K)。

**实现要点**：
```python
# B 物理 shape (N, K)，加载逻辑 (K, N) 即 B^T
b_block_ptr = tl.make_block_ptr(
    base=b_ptr,
    shape=(N, K),
    strides=(stride_bn, stride_bk),
    offsets=(pid_n * BLOCK_N, 0),
    block_shape=(BLOCK_N, BLOCK_K),
    order=(1, 0),
)
for _ in range(0, K, BLOCK_K):
    b = tl.load(b_block_ptr, boundary_check=(0, 1))
    ...
    b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_K))
```
- `BLOCK_K` 固定为 256，满足 512B；
- `BLOCK_N` 取 128/256；
- `M0/N0` 可按实际 shape 灵活调整，但 K0 不动。

**常见坑**：
- 把 B 转置后的 stride 搞反（`stride_bk` 与 `stride_bn` 混淆）；
- K 循环推进方向与 B 的物理 K 维不一致；
- 尾部 K 块未做 padding 导致越界。

### 5. 双转置场景：BLOCK_N 适配转置后数据排布

**问题根因**：`trans_a=True, trans_b=True` 时，A 物理 shape (K, M)，B 物理 shape (N, K)。逻辑上 C = A^T @ B^T，两个操作数的“K 维”在物理中分别是 A 的外维和 B 的内维，分块需要同时保证两边连续访问。

**收益明显条件**：
- `trans_a=True, trans_b=True`；
- M、N、K 均 ≥256；
- 需要专门路径，不能复用单转置配置。

**实现要点**：
```python
# A 物理 (K, M)，加载逻辑 (M, K)
a_block_ptr = tl.make_block_ptr(
    base=a_ptr,
    shape=(K, M),
    strides=(stride_ak, stride_am),
    offsets=(0, pid_m * BLOCK_M),
    block_shape=(BLOCK_K, BLOCK_M),
    order=(1, 0),
)
# B 物理 (N, K)，加载逻辑 (K, N)
b_block_ptr = tl.make_block_ptr(
    base=b_ptr,
    shape=(N, K),
    strides=(stride_bn, stride_bk),
    offsets=(pid_n * BLOCK_N, 0),
    block_shape=(BLOCK_N, BLOCK_K),
    order=(1, 0),
)
```
- `M0=256`、`K0=256` 满足 512B；
- `N0=128` 适配转置后的数据排布；
- 使用 `tl.trans` 或 stride 直接构造 `tl.dot` 所需 layout。

**常见坑**：
- 双转置时仍用单转置的 block_ptr 配置，导致逻辑乘法维度不匹配；
- 忘记对加载后的 tile 做 `tl.trans`；
- `BLOCK_N=128` 虽满足 512B，但若 N 不是 128 倍数需尾部处理。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_matmul_003` | 10% | A、B 均不转置的 matmul：分块行宽严格对齐 512B，使用列优先 PID 分组提升 L2 命中。 |
| `exp_matmul_004` | 10% | A 不转置、B 转置的 matmul：`BLOCK_K` 固定为 256 满足 512B，`BLOCK_M/BLOCK_N` 按实际 shape 灵活调整。 |
| `exp_matmul_005` | 32% | A、B 均转置的 matmul：专用分块 `M0=256, K0=256, N0=128`，同时适配 A(K,M) 与 B(N,K) 的物理 stride。 |

## 代表 Kernel

选择 `exp_matmul_005` 作为代表（收益 32%），因为它覆盖最复杂的 **双转置分块** 场景，且对 512B 对齐和 block_ptr 使用要求最严格。下面为合成的双转置 matmul kernel，展示 `BLOCK_M=256, BLOCK_K=256, BLOCK_N=128` 配置与 `tl.make_block_ptr` 用法：

```python
import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_SIZE': 16},
            num_stages=2, num_warps=4,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_double_transpose_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,  # A 物理 (K, M)
    stride_bk, stride_bn,  # B 物理 (N, K)
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # 列优先 PID 分组
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # A 物理 (K, M)，加载逻辑 A^T (M, K)
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(K, M),
        strides=(stride_ak, stride_am),
        offsets=(0, pid_m * BLOCK_M),
        block_shape=(BLOCK_K, BLOCK_M),
        order=(1, 0),
    )
    # B 物理 (N, K)，加载逻辑 B^T (K, N)
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))  # (BLOCK_K, BLOCK_M)
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))  # (BLOCK_N, BLOCK_K)
        # 逻辑上 A^T @ B^T：a_tile.T (M,K) @ b_tile.T (K,N)
        acc += tl.dot(tl.trans(a_tile), tl.trans(b_tile))
        a_block_ptr = tl.advance(a_block_ptr, (BLOCK_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_K))

    c = acc.to(tl.float16)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def matmul_double_transpose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (K, M), b: (N, K), output c: (M, N)"""
    K, M = a.shape
    N, K2 = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_double_transpose_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
```

**关键实现细节**：
- `BLOCK_N=128`、fp16 下行宽 256B？注意：双转置经验中 `N0=128` 指物理 N 维，fp16 128 元素 = 256B；若要求 512B，通常需配合 `BLOCK_K=256` 或整体 tile 满足 512B。实际经验中的 `N0=128` 是在双转置特殊排布下经测试选出的最优值，使用时需验证后端是否触发 partial read。
- A/B 的 `shape`、`strides`、`order` 都对应**物理张量**，而非逻辑 matmul 的 `M×K`/`K×N`；
- K 循环中 A 沿物理 K 维（shape 第 0 维）推进，B 沿物理 K 维（shape 第 1 维）推进；
- 加载后通过 `tl.trans` 得到 `tl.dot` 所需的 `(M,K)` 和 `(K,N)`；
- 累加器用 fp32，最后 cast 到 fp16。

## 已知实例

- `exp_matmul_003` — Split（收益 10%）
  - A、B 均不转置的 matmul：分块行宽严格对齐 Ascend 后端 512B 整数倍要求，最大化带宽利用。
- `exp_matmul_004` — Split（收益 10%）
  - A 不转置、B 转置的 matmul：`K0` 固定为 256 满足 512B 要求，`M0/N0` 可按实际矩阵尺寸灵活调整。
- `exp_matmul_005` — Split（收益 32%）
  - A、B 均转置的 matmul：`M0=256`、`K0=256` 满足 512B 对齐，`N0=128` 适配转置后的数据排布。代表 kernel 见上。

## 关联知识

- 开发指南: [MatMul / GEMM 高性能算子](../../reference/op-class-matmul.md)
- 优化原则: [Cube-Vector 场景行宽必须 512B 对齐](../memory-sram/row-width-512b-align.md)
- 优化原则: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../memory-sram/block-ptr-vs-scalar-load.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [MatMul / GEMM 高性能算子](../../reference/op-class-matmul.md) — 主题相关：共享 910b2, 910b3, matmul, tiling
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 matmul
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 matmul
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](../../ops/fused-floyd-attention-grad.md) — 主题相关：共享 matmul
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 matmul
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 matmul
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 matmul
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 matmul
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 matmul
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 matmul
- 相关主题: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](../algorithm-numerics/dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 相关主题: [编译器融合消除中间 dtype 舍入时，拆分独立 kernel 强制 GM round-trip](../algorithm-numerics/split-kernel-gm-roundtrip.md) — 同属优化知识，主题相关
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [FlashAttention 变体的内存访问优化](../algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, block_pointer
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](../memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](../memory-sram/traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, block_pointer
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../memory-sram/traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, transpose
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](../memory-sram/traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, matmul, tiling
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Matmul 硬件适配优化聚合](../portability/traj-matmul-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, block_pointer, tiling
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [Matmul Scheduling 优化聚合](traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, matmul

<!-- okf:related:end -->
