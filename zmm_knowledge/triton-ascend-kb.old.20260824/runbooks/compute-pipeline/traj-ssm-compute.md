---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: ssm
dimension: "2.1"
title: SSM Compute 优化聚合
description: 聚合 8 条 SSM / 状态空间模型在 Ascend 910B 上的 compute 优化经验，覆盖 Delta Rule 融合、Chunk 化扫描、标量循环改矩阵乘与结构级优化。
tags: [ssm, compute, delta_rule, mamba, state_space_model, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/ssm/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# SSM Compute 优化聚合

## 概述

状态空间模型（SSM，如 Mamba、RWKV、Delta Rule 系列）的核心计算不是密集的矩阵乘，而是**递推状态更新、chunk 化扫描和门控融合**。在 Ascend 910B 上，这些 kernel 容易受困于标量指令、频繁同步和低 Cube 利用率。本卡聚合 8 条 compute 方向的 experience，覆盖从单 kernel 结构优化到算法级重构的典型模式，收益 25%–50%。

这些经验的共同前提：
- 序列通常很长，但 hidden 维度可控；
- 状态矩阵需要在多个时间步/ chunk 间复用；
- 数值稳定性（softplus、gate 归一化）必须和性能同时保证。

## 共同模式

### 1. Delta Rule 状态更新融合

Delta Rule 类 SSM 把离散化、门控、状态更新合并到一个 kernel 中，避免中间张量反复写回 HBM。关键设计：
- 状态矩阵在寄存器/SRAM 中保持，按 chunk 更新；
- softplus/sigmoid 门控在 fp32 上计算；
- 混合精度累加：中间状态 fp32，输出按模型精度 downcast。

收益来源：减少 kernel launch、隐藏内存延迟、提高状态复用率。

### 2. Chunk 化扫描

把长序列切分为固定大小的 chunk（如 64/128），每个 chunk 内部先做局部扫描，再处理 chunk 间的传递状态。这能：
- 把长递推拆成可并行的块级计算；
- 让状态矩阵常驻寄存器，减少全局内存访问；
- 通过 `cu_seqlens` 支持变长序列。

典型应用：chunk local cumsum、chunk K*K^T、chunk forward。

### 3. 标量循环改写为矩阵乘

SSM backward 中常见“标量 × 向量 broadcast”的逐元素计算，这会严重低估 Cube 利用率。把它改写为显式矩阵乘后：
- 计算密度从 O(n) 提升到 O(n²) 块级运算；
- 编译器可以生成 Cube dot 指令；
- 需要注意 gate 归一化点的数值稳定性。

### 4. Kernel 结构拆分与循环不变量外提

复杂 SSM backward kernel（如 KDA）通常包含多个 phase 和大量 barrier。结构级优化包括：
- 按计算依赖把 kernel 拆为多个 phase，减少不必要同步；
- 把 barrier 提到循环外；
- 预计算 stride 和 `K % BK == 0` 快速路径；
- 循环不变量外提，减少标量地址计算。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_kda_bwd_wy_structure_opt_001` | 30% | KDA backward 结构优化：phase 拆分、barrier 外提、stride 预计算 |
| `exp_fused_delta_rule_001` | 35% | Delta Rule 状态更新融合，BK/BV 适配 SRAM |
| `exp_chunk_cumsum_001` | 25% | Chunk 化局部 cumsum，支持 cu_seqlens 变长 |
| `exp_chunk_kkt_001` | 35% | Chunk 化 K*K^T 计算，block_ptr + 向量化 mask |
| `exp_fused_delta_rule_npu_001` | 35% | NPU 特化 Delta Rule，寄存器级状态驻留 |
| `exp_ssm_gated_delta_001` | 50% | Selective SSM gated delta rule，2D tiling + 原地状态更新 |
| `exp_kda_bwd_scalar_loop_to_dot_001` | 40% | 标量×向量 broadcast 改矩阵乘，提升 Cube 利用率 |
| `exp_chunk_fwd_001` | 35% | Chunk forward，block_ptr tiling + 混合精度累加 |

## 代表 Kernel

以下为 `exp_ssm_gated_delta_001` 对应的 kernel，收益 50%，集中体现了 **Delta Rule 融合 + 2D tiling + 寄存器级状态驻留**：

```python
@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, o_ptr, h_ptr,
    batch_size, n_heads, seq_len, d_k, d_v,
    stride_qb, stride_qh, stride_qt, stride_qk,
    stride_kb, stride_kh, stride_kt, stride_kk,
    stride_vb, stride_vh, stride_vt, stride_vv,
    stride_gb, stride_gh, stride_gt,
    stride_ob, stride_oh, stride_ot, stride_ov,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # 每个 (batch, head) 维护一个 d_k × d_v 状态矩阵
    h = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    for t in range(0, tl.cdiv(seq_len, BLOCK_T)):
        offs_t = t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)

        q = tl.load(q_ptr + pid_b * stride_qb + pid_h * stride_qh +
                    offs_t[:, None] * stride_qt + offs_k[None, :] * stride_qk,
                    mask=(offs_t[:, None] < seq_len) & (offs_k[None, :] < d_k))
        k = tl.load(k_ptr + pid_b * stride_kb + pid_h * stride_kh +
                    offs_t[:, None] * stride_kt + offs_k[None, :] * stride_kk,
                    mask=(offs_t[:, None] < seq_len) & (offs_k[None, :] < d_k))
        v = tl.load(v_ptr + pid_b * stride_vb + pid_h * stride_vh +
                    offs_t[:, None] * stride_vt + offs_v[None, :] * stride_vv,
                    mask=(offs_t[:, None] < seq_len) & (offs_v[None, :] < d_v))
        g = tl.load(g_ptr + pid_b * stride_gb + pid_h * stride_gh +
                    offs_t[:, None] * stride_gt,
                    mask=offs_t[:, None] < seq_len)

        # Delta rule: h = diag(g) @ h + k^T @ v,  o = q @ h
        g = tl.exp(g).to(tl.float32)
        h = h * g[None, :]  # 简化示意：实际按维度广播
        kt_v = tl.dot(tl.trans(k), v, allow_tf32=False)
        h = h + kt_v
        o = tl.dot(q, h, allow_tf32=False)

        tl.store(o_ptr + pid_b * stride_ob + pid_h * stride_oh +
                 offs_t[:, None] * stride_ot + offs_v[None, :] * stride_ov,
                 o.to(o_ptr.dtype.element_ty),
                 mask=(offs_t[:, None] < seq_len) & (offs_v[None, :] < d_v))

    # 可选：把最终状态写回 h_ptr 供下一 chunk 使用
    tl.store(h_ptr + pid_b * d_k * d_v + pid_h * d_k * d_v +
             tl.arange(0, BLOCK_K)[:, None] * d_v + tl.arange(0, BLOCK_V)[None, :],
             h.to(h_ptr.dtype.element_ty),
             mask=(tl.arange(0, BLOCK_K)[:, None] < d_k) &
                  (tl.arange(0, BLOCK_V)[None, :] < d_v))
```

关键实现细节：
- 状态矩阵 `h` 全程驻留寄存器，跨时间步复用；
- `allow_tf32=False` 保证状态更新精度；
- gate `g` 用 fp32 计算 exp，避免数值不稳定；
- 通过 BLOCK_T/BLOCK_K/BLOCK_V 二维 tiling 适配 SRAM 容量。

## 已知实例

- `exp_kda_bwd_wy_structure_opt_001` — Compute（收益 30%）
  - KDA backward 结构级优化：phase 拆分、barrier 外提、循环不变量外提、stride 预计算、K%BK==0 快速路径。
- `exp_fused_delta_rule_001` — Compute（收益 35%）
  - Delta Rule 状态更新融合，BK/BV tile 大小需适配 SRAM/寄存器容量。
- `exp_chunk_cumsum_001` — Compute（收益 25%）
  - Chunk 化局部 cumsum，BLOCK_T 需是 CHUNK_SIZE 整数倍，支持 cu_seqlens 变长。
- `exp_chunk_kkt_001` — Compute（收益 35%）
  - Chunk 化 K*K^T 计算，使用 block_ptr 抽象、tiling 累加、向量化 mask。
- `exp_fused_delta_rule_npu_001` — Compute（收益 35%）
  - NPU 特化 Delta Rule，深度 kernel 融合、寄存器级状态驻留、数值稳定 softplus。
- `exp_ssm_gated_delta_001` — Compute（收益 50%）
  - Selective SSM 门控 Delta Rule，2D tiling + 原地状态更新，代表 kernel 见上。
- `exp_kda_bwd_scalar_loop_to_dot_001` — Compute（收益 40%）
  - 把标量×向量 broadcast 改为矩阵乘，提升 Cube 利用率，需注意 gate 归一化精度。
- `exp_chunk_fwd_001` — Compute（收益 35%）
  - Chunk forward，block_ptr tiling、门控 mask 融合、混合精度累加。

## 关联知识

- 开发指南: [SSM / 状态空间模型算子优化](../../reference/op-class-ssm.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [卷积算子优化（Causal Conv1D）](../../reference/op-class-convolution.md) — 主题相关：共享 910b2, 910b3, mamba, ssm
- 对应指南: [SSM / 状态空间模型算子优化](../../reference/op-class-ssm.md) — 主题相关：共享 910b2, 910b3, delta_rule, mamba, ssm
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 ssm
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [重叠 T-tiling 保证跨 tile 滑动窗口连续性](../memory-sram/overlapping-t-tiling.md) — 主题相关：共享 ssm
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](../memory-sram/register-shift-sliding-window.md) — 主题相关：共享 ssm
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](../memory-sram/traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, ssm
- 相关主题: [标量循环改写为 tl.dot，提升 Cube 利用率](scalar-loop-to-dot.md) — 主题相关：共享 ssm
- 相关主题: [SwiGLU 计算优化：门控融合、展平布局与激活量化](traj-activation-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Causal Conv1D Compute 优化聚合](traj-conv-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, mamba, ssm
- 相关主题: [Scatter Weight Gradient 向量化计算优化](traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Loss 计算优化聚合](traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [MoE 稀疏 Weight Gradient 计算优化](traj-moe-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Reduction Compute 优化聚合](traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, compute, ssm
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](traj-solve_tril-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [SSM 反向 kernel 标量消除：i64 降级、整数比较与 boundary_check 优化](traj-ssm-scalar_elimination.md) — 主题相关：共享 910b2, 910b3, ssm

<!-- okf:related:end -->
