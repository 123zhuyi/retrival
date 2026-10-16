---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: matmul
dimension: "1.1"
title: Matmul 内存访问优化：分块策略、L0 复用与算子融合
description: 聚合 2 条 matmul 相关在 Ascend 910B 上的 memory access 经验，覆盖大矩阵分块 tiling、L0/L0C 中间结果复用，以及 residual add + RMSNorm 的算子融合与 weight hoisting，收益 35%–40%。
tags: [matmul, memory_access, tiling, l0_reuse, operator_fusion, weight_hoisting, rmsnorm, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/matmul/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Matmul 内存访问优化：分块策略、L0 复用与算子融合

## 概述

本卡虽然归类在 matmul 下，但 2 条 experience 分别来自两类不同场景：
1. **大矩阵乘法（GEMM）**：核心是让分块大小匹配 Ascend Cube 的 L0 访存粒度，并把累加器驻留在 L0C/寄存器中复用；
2. **Gemma RMSNorm 融合**：核心是把 residual add 与 RMSNorm 合成一个 kernel，并把共享的 weight 提到循环外，减少 HBM 读写。

两者共同点是**通过减少不必要的内存搬运来提升有效带宽利用率**——matmul 减少 L0 与 HBM 之间的数据往返，RMSNorm 融合消除中间张量写入。

**核心 trade-off**：
- 大 tile 提升数据复用，但寄存器/L0 压力增大；小 tile 灵活但复用率低；
- 算子融合省内存，但会把多个计算阶段耦合在同一个 kernel 内，可能限制调度；
- weight hoisting 依赖 weight 在循环间不变，且需要足够寄存器缓存。

## 共同模式

### 1. L0 级分块匹配 Cube 粒度

**问题根因**：Ascend Cube 对 L0 层访存有固定粒度偏好。若 BLOCK_M/BLOCK_N/BLOCK_K 与硬件粒度不匹配，会导致微片访问、bank conflict 或 L0 利用率不足。

**收益明显条件**：
- 大矩阵 `m, n, k >= 256`（甚至 >= 1024）；
- 数据类型为 fp16/bf16，可被 Cube 高效处理；
- 输入布局无需转置或重排。

**实现要点**：
- 大矩阵用 256×256×256 分块，最大化每个 tile 的计算量；
- 中小矩阵用 128×128×128，避免 tile 太大导致 spill；
- 保持 `BLOCK_K` 与 L0 缓存行对齐，提升 A/B tile 加载效率。

**常见坑**：
- 不要对所有尺寸都用 256 tile，小矩阵会空载或 spill；
- 边界 mask 要处理 `M - rm*BLOCK_M` 和 `N - rn*BLOCK_N` 的余量；
- 输入 stride 必须按实际张量传递，避免非合并访问。

### 2. L0C/寄存器累加器复用

**问题根因**：`tl.dot` 的累加结果如果频繁写回 HBM 再读入，会浪费大量 L0 ↔ HBM 带宽。把 `acc` 保留在 L0C 或寄存器中直到 K 维遍历结束，可显著减少内存流量。

**收益明显条件**：
- K 维远大于 BLOCK_K，需要多次迭代；
- 累加器数据类型为 fp32，占据寄存器/L0C；
- 算子本身以 matmul 为主。

**实现要点**：
```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptr, mask=..., other=0.0)
    b = tl.load(b_ptr, mask=..., other=0.0)
    acc += tl.dot(a, b)
    a_ptr += BLOCK_K
    b_ptr += BLOCK_K * N
# K 维遍历完再一次性写回
acc = acc.to(x.dtype)
tl.store(c_ptr, acc, mask=...)
```
- 所有中间累加在 fp32 完成；
- 仅在 K 循环结束后 downcast 并 store。

**常见坑**：
- `acc` 在寄存器中，BLOCK_M×BLOCK_N 太大时会 spill；
- 不要在一个 K 迭代内把 `acc` 写回再读入；
- 注意 `tl.dot` 的输入数据类型需匹配 Cube 要求。

### 3. 算子融合消除中间张量

**问题根因**：`residual add` 和 `RMSNorm` 若分两个 kernel 执行，会把 `hidden_state + residual` 的中间结果写回 HBM，再读入下一个 kernel，产生两次完整张量的 HBM 流量。

**收益明显条件**：
- 连续多个 elementwise/reduction 算子作用于同一数据；
- 中间张量没有下游其他消费者；
- batch 和 dim 足够大，能占满内存带宽。

**实现要点**：
- 把 `x + residual` 与 `RMSNorm(x+residual, weight)` 写在一个 kernel；
- add 结果直接驻留寄存器，作为 RMSNorm 的输入；
- 最终 norm 输出直接 store，无需中间 buffer。

**常见坑**：
- 融合后 kernel 寄存器压力增大，需评估是否 spill；
- 如果中间张量需要被反向传播复用，则不能融合；
- Gemma 的 RMSNorm 有 `w + 1.0` 特殊处理，需与标准 RMSNorm 区分。

### 4. Weight Hoisting：循环外加载共享 weight

**问题根因**：RMSNorm 的 weight 对同一 batch 内所有 row 都相同。如果把 `w = tl.load(weight_ptr + offset_d)` 放在 `for row_start` 循环内部，每个 row 都重复加载同一段 weight。

**收益明显条件**：
- weight 在循环维度上不变；
- 循环迭代次数多（batch 大）；
- weight 尺寸不大，可被寄存器/缓存缓存。

**实现要点**：
```python
# 在 row 循环外加载一次 weight
w = tl.load(weight_ptr + offset_d).to(tl.float32)

for row_start in tl.range(start_batch, end_batch, BLOCK_M):
    x = tl.load(hidden_state_ptr + offset_hidden, mask=mask_bs)
    residual = tl.load(residual_ptr + offset_hidden, mask=mask_bs)
    add_val = x + residual
    # ... RMSNorm 计算复用同一个 w ...
```
- weight 提升到循环外；
- 若 dim 较大，可进一步按 BLOCK_D 分块 hoist。

**常见坑**：
- weight 提升后若寄存器不足，会 spill 到 L1 甚至 HBM，反而更慢；
- 若 weight 在不同 row 需要不同处理（如 per-channel scale 变化），不能 hoist；
- 注意 cast 到 fp32 的时机，避免在循环内重复转换。

### 5. 混合精度：fp32 累加 + fp16/bf16 搬运

**问题根因**：低精度 load/store 提升带宽，但 reduction/累加需要 fp32 保证数值稳定。

**收益明显条件**：
- 数据类型为 fp16/bf16；
- 包含 `tl.sum`、`tl.dot`、variance 等累加操作；
- 对精度敏感（如训练、RMSNorm）。

**实现要点**：
- load 后立即 `.to(tl.float32)`；
- 所有 reduction 和 dot 在 fp32 完成；
- store 前再 downcast 到原 dtype。

**常见坑**：
- 不要为了提高速度把 `tl.sum` 直接在 fp16 中做，会严重累积误差；
- downcast 时机太晚可能增加寄存器占用；
- 某些算子对 fp32 中间值的 store 有特殊布局要求。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_gemma_rmsnorm_001` | 35% | residual add + RMSNorm 融合为一个 kernel，weight 提到 row 循环外，fp32 累加方差，Gemma 风格 `w + 1.0` 缩放。 |
| `exp_matmul_001` | 40% | 大尺寸 GEMM（m/n/k≥256）采用 256×256×256 分块，累加器保留在 L0C/fp32，K 维遍历完再写回；小矩阵建议 128 分块。 |

## 代表 Kernel

选择 `exp_matmul_001` 作为代表（收益 40%），因为它最直接体现 **matmul 内存访问优化** 主题：分块 tiling、L0C 累加器复用、延迟写回。以下为基于该 experience 的示意：

```python
@triton.jit
def matmul_kernel(
    x_ptr, y_ptr, output_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid // (N // BLOCK_N)
    rn = pid % (N // BLOCK_N)

    # A tile 指针：x[rm*BLOCK_M : (rm+1)*BLOCK_M, 0:BLOCK_K]
    a_ptr = x_ptr + rm * BLOCK_M * K \
            + tl.arange(0, BLOCK_M)[:, None] * K \
            + tl.arange(0, BLOCK_K)[None, :]
    # B tile 指针：y[0:BLOCK_K, rn*BLOCK_N : (rn+1)*BLOCK_N]
    b_ptr = y_ptr + rn * BLOCK_N \
            + tl.arange(0, BLOCK_K)[:, None] * N \
            + tl.arange(0, BLOCK_N)[None, :]

    # L0C/寄存器累加器，fp32 精度
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptr,
            mask=tl.arange(0, BLOCK_K)[None, :] < K - k,
            other=0.0,
        )
        b = tl.load(
            b_ptr,
            mask=tl.arange(0, BLOCK_K)[:, None] < K - k,
            other=0.0,
        )
        acc += tl.dot(a, b)
        a_ptr += BLOCK_K
        b_ptr += BLOCK_K * N

    # K 维遍历结束后一次性写回
    acc = acc.to(x_ptr.dtype.element_ty)
    output_ptr = output_ptr + rm * BLOCK_M * N \
                 + tl.arange(0, BLOCK_M)[:, None] * N \
                 + tl.arange(0, BLOCK_N)[None, :]
    tl.store(
        output_ptr,
        acc,
        mask=(tl.arange(0, BLOCK_M)[:, None] < M - rm * BLOCK_M) \
           & (tl.arange(0, BLOCK_N)[None, :] < N - rn * BLOCK_N),
    )
```

**关键实现细节**：
- `acc` 全程 fp32，直到最后才 downcast；
- K 维循环内只加载 A/B tile、做一次 `tl.dot`，不 store；
- 边界 mask 同时覆盖 M 和 N 两个方向的余量；
- 大尺寸用 256 tile，中小尺寸建议 128 tile 以避免 spill。

对应 `exp_gemma_rmsnorm_001` 的融合 + weight hoisting 片段：

```python
@triton.jit
def add_gemma_rms_norm_kernel(
    hidden_state_ptr, hidden_state_stride_bs,
    weight_ptr,
    residual_ptr,
    add_output_ptr, norm_output_ptr,
    variance_epsilon,
    batch, dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    core_id = tl.program_id(0)
    core_num = tl.num_programs(0)
    batch_per_core = tl.cdiv(batch, core_num)
    start_batch = core_id * batch_per_core
    end_batch = tl.minimum(start_batch + batch_per_core, batch)
    offset_d = tl.arange(0, dim)

    # weight hoisting：在 row 循环外加载一次 weight
    w = tl.load(weight_ptr + offset_d).to(tl.float32)

    for row_start in tl.range(start_batch, end_batch, BLOCK_M):
        offset_row = row_start + tl.arange(0, BLOCK_M)
        offset_hidden = offset_row[:, None] * hidden_state_stride_bs + offset_d[None, :]
        mask_bs = offset_row[:, None] < batch

        x = tl.load(hidden_state_ptr + offset_hidden, mask=mask_bs)
        residual = tl.load(residual_ptr + offset_hidden, mask=mask_bs)
        add_val = x + residual
        tl.store(add_output_ptr + offset_hidden, add_val, mask=mask_bs)

        # 融合：add 结果直接做 RMSNorm，无需写回 HBM 再读入
        x_fp32 = add_val.to(tl.float32)
        variance = tl.sum(x_fp32 * x_fp32, axis=-1) / dim
        x_fp32 = x_fp32 * tl.rsqrt(variance[:, None] + variance_epsilon)
        # Gemma 风格：weight 加 1.0
        x_fp32 = x_fp32 * (w + 1.0)
        output = x_fp32.to(x.dtype)
        tl.store(norm_output_ptr + offset_hidden, output, mask=mask_bs)
```

**关键实现细节**：
- `w` 在 row 循环外加载，避免每个 row 重复读取 weight；
- residual add 的结果 `add_val` 直接作为 RMSNorm 输入，省一次 HBM 往返；
- variance 计算在 fp32 完成，保证数值稳定；
- Gemma 的 `w + 1.0` 与标准 RMSNorm 的 `w` 缩放不同，实现时需注意。

## 已知实例

- `exp_gemma_rmsnorm_001` — Memory Access（收益 35%）
  - residual add + RMSNorm 融合为一个 kernel，weight 提到 row 循环外，fp32 累加方差，Gemma 风格 `w + 1.0` 缩放。
- `exp_matmul_001` — Memory Access（收益 40%）
  - 大尺寸 GEMM 采用 256×256×256 分块，累加器保留在 L0C/fp32，K 维遍历完再写回；小矩阵建议 128 分块，代表 kernel 见上。

## 关联知识

- 开发指南: [MatMul / GEMM 高性能算子](../../reference/op-class-matmul.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [融合算子综合页](../../reference/op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, rmsnorm
- 对应指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 910b2, 910b3, rmsnorm
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 matmul, rmsnorm
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 matmul, rmsnorm
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 matmul, rmsnorm
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 rmsnorm
- 相关主题: [Rmsnorm Fused Skeleton](../../reference/skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 rmsnorm
- 相关主题: [FlashAttention 变体的内存访问优化](../algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, operator_fusion, rmsnorm
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, rmsnorm, tiling
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](../compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 rmsnorm
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, tiling
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](../compute-pipeline/traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, operator_fusion
- 相关主题: [Matmul Compute 优化聚合](../compute-pipeline/traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, matmul, operator_fusion
- 相关主题: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](../compute-pipeline/traj-matmul-split.md) — 主题相关：共享 910b2, 910b3, matmul, tiling
- 相关主题: [Misc 算子 Scheduling 优化聚合](../compute-pipeline/traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, operator_fusion
- 相关主题: [Normalization Scheduling 优化聚合](../compute-pipeline/traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, rmsnorm
- 相关主题: [Reduction Compute 优化聚合](../compute-pipeline/traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, rmsnorm
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [Matmul 硬件适配优化聚合](../portability/traj-matmul-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 主题相关：共享 memory_access
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](row-width-512b-align.md) — 主题相关：共享 matmul
- 相关主题: [Activation Backward 内存访问优化聚合](traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Gather/Scatter 内存访问优化聚合](traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Misc 算子内存访问优化聚合](traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Sum-and-Scale 归约内存访问优化](traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Softmax Backward 内存访问优化聚合](traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](traj-solve_tril-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access

<!-- okf:related:end -->
