---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: matmul
dimension: "2.1"
title: Matmul Compute 优化聚合
description: 聚合 10 条 matmul 在 Ascend 910B 上的 compute 优化经验，覆盖 RoPE/偏置/scale 融合、量化、向量并行、块矩阵求逆与小矩阵特化。
tags: [matmul, gemm, compute, operator_fusion, activation_fusion, quantization, vector_parallelism, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/matmul/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Matmul Compute 优化聚合

## 概述

Matmul 的 compute 优化目标是在保证数值稳定的前提下，**让 Cube 单元持续饱和、减少冗余的 Vector/Cube 数据往返、并把线性层前后的小算子融合进主循环**。本卡聚合 10 条 matmul 类别下的 compute 经验，覆盖 Transformer 线性层、QK projection、偏置融合、块矩阵求逆等场景，收益 15%–35%。

这些经验的共同特点：
- 计算密集型，但常因 activation/bias/scale 等小操作产生额外 kernel launch；
- 需要精细的精度控制（fp32 累加、allow_tf32=False、量化对齐）；
- 部分场景（小矩阵、特殊结构）需要偏离标准 GEMM tiling。

## 共同模式

### 1. 矩阵乘与前后算子融合

将 Matmul 与紧邻的 RoPE、Bias、Scale、RMSNorm 等操作合并到一个 kernel 中，避免把中间结果写回 HBM。典型组合：
- RoPE + Matmul（Q/K projection）；
- Matmul + Bias（标准线性层）；
- Matmul + Scale（LayerNorm 后的缩放）；
- MHC pre-GEMM：先 reduction/sqrsum 再 GEMM。

收益来源：消除中间张量的 HBM 读写和额外 kernel launch。

### 2. 量化与精度控制

大模型推理常用低精度激活/权重，但矩阵乘累加必须保持 fp32，否则累积误差会放大。关键做法：
- `tl.dot(..., allow_tf32=False)` 强制 FP32 累加；
- 量化融合：反量化与 Matmul 合并，避免单独反量化 kernel；
- 输入/输出按硬件要求的 block 对齐（如 512B 行宽）。

### 3. 向量并行 + Activation Fusion

利用 `dl.parallel` 等硬件方言，把元素级 activation（如 SwiGLU、GELU）与 Matmul 并行执行或流水线执行。这要求：
- activation 的计算量足够小，不成为瓶颈；
- 数据布局允许 Vector 单元直接消费 Matmul 输出；
- 通过 autotune 选择平衡的 BLOCK_M/BLOCK_N。

### 4. 特殊结构与部分矩阵乘

- **块矩阵求逆**：利用 16×16/32×32 block 结构和 `make_block_ptr`，把 off-diagonal block 计算映射到 Cube；
- **Partial matmul**：对 K 不对齐或分片矩阵，用 `grouped_launch_diagonal` 动态映射 block，配合 `dot_pad_only_k`；
- **小矩阵**：M,N,K ≤ 64 时使用 16×16×16 tile，避免大 tile 的 mask 和填充浪费。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_rope_matmul_fused_001` | 25% | RoPE + Matmul 融合，预计算 sin/cos，双路累加 |
| `exp_cube1_qk_gemm_001` | 35% | Attention QK projection，NC1HWC0 友好加载，Cube 加速 |
| `exp_mhc_fused_gemm_sqrsum_001` | 30% | MHC 前置 reduction + GEMM 融合，hc_mult3 ≤ 32 场景 |
| `exp_partial_matmul_001` | 25% | 部分/分片矩阵乘，对角 block 映射 + K-only padding |
| `exp_matmul_002` | 35% | 量化融合 + 循环展开 + 内存预取 |
| `exp_matmul_vec_parallel_001` | 35% | 向量并行 + activation fusion + GROUP_M L2 复用 |
| `exp_matmul_triton_base_001` | 15% | 小矩阵基线：16×16×16 tile，allow_tf32=False |
| `exp_matmul_bias_fused_001` | 30% | Matmul + Bias 融合，persistent kernel，flexible stride |
| `exp_fused_gemm_scale_001` | 25% | GEMM + Scale 融合，fp32 累加，边界 mask |
| `exp_block_matmul_inv_001` | 35% | 块矩阵求逆 off-diagonal 块，16×16 sub-block + Tensor Core |

## 代表 Kernel

以下为 `exp_matmul_vec_parallel_001` 对应的 kernel，收益 35%，集中体现了**向量并行 + activation fusion + L2 复用**三个模式：

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(c_ptr.dtype.element_ty)
    if ACTIVATION == "swiglu":
        c = swiglu(c)
    elif ACTIVATION == "gelu":
        c = gelu(c)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
```

关键实现细节：
- `GROUP_M` 重排 program ID，提升 A 矩阵在 L2 中的复用；
- `allow_tf32=False` 保证 fp32 累加精度；
- activation 在寄存器内完成，避免写回 HBM；
- 通过 `ACTIVATION: tl.constexpr` 编译期开关控制融合路径。

## 已知实例

- `exp_rope_matmul_fused_001` — Compute（收益 25%）
  - RoPE 与 Matmul 融合，预计算 sin/cos，双路累加，适用于 Transformer 推理中 Q/K projection。
- `exp_cube1_qk_gemm_001` — Compute（收益 35%）
  - Attention QK projection 专用 kernel，采用 NC1HWC0 友好加载，使用 `cube1` 硬件原语加速。
- `exp_mhc_fused_gemm_sqrsum_001` — Compute（收益 30%）
  - MHC 前置 reduction 与 GEMM 融合，针对 hc_mult3 ≤ 32 的小维度特征预处理。
- `exp_partial_matmul_001` — Compute（收益 25%）
  - 部分矩阵乘，K 不对齐或分片矩阵场景，使用对角 block 映射和 K-only padding。
- `exp_matmul_002` — Compute（收益 35%）
  - 量化融合：反量化与 Matmul 合并，配合循环展开和内存预取。
- `exp_matmul_vec_parallel_001` — Compute（收益 35%）
  - 向量并行 + activation fusion + GROUP_M 提升 L2 复用，代表 kernel 见上。
- `exp_matmul_triton_base_001` — Compute（收益 15%）
  - 小矩阵基线实现，16×16×16 tile，allow_tf32=False，适合 M,N,K ≤ 64。
- `exp_matmul_bias_fused_001` — Compute（收益 30%）
  - Matmul + Bias 融合，persistent kernel 风格，支持非连续 stride。
- `exp_fused_gemm_scale_001` — Compute（收益 25%）
  - GEMM + Scale 融合，Transformer 线性层后常见，fp32 累加后 downcast。
- `exp_block_matmul_inv_001` — Compute（收益 35%）
  - 块矩阵求逆的 off-diagonal 块计算，16×16 sub-block 结构，适配 Tensor Core。

## 关联知识

- 开发指南: [Matmul / GEMM 算子优化](../../reference/op-class-matmul.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [融合算子综合页](../../reference/op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, gemm
- 对应指南: [MatMul / GEMM 高性能算子](../../reference/op-class-matmul.md) — 主题相关：共享 910b2, 910b3, gemm, matmul
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 matmul
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 matmul
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](../../ops/fused-floyd-attention-grad.md) — 主题相关：共享 matmul
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 matmul
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 matmul
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 matmul
- 相关主题: [Gemm Basic Skeleton](../../reference/skeleton-gemm_basic.md) — 同属优化知识，主题相关
- 相关主题: [GEMM 操作数保持 low_dtype，fp32 累加](../algorithm-numerics/gemm-low-dtype-fp32-accum.md) — 主题相关：共享 gemm
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, compute, operator_fusion
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](../memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](../memory-sram/traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, matmul, operator_fusion
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../meta-skills/quant-fusion-int8-precision-diagnosis.md) — 主题相关：共享 quantization
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Matmul 硬件适配优化聚合](../portability/traj-matmul-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 910b2, 910b3, activation_fusion, gemm, matmul, vector_parallelism
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 gemm, quantization
- 相关主题: [SwiGLU 计算优化：门控融合、展平布局与激活量化](traj-activation-compute.md) — 主题相关：共享 910b2, 910b3, compute, quantization
- 相关主题: [Causal Conv1D Compute 优化聚合](traj-conv-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Scatter Weight Gradient 向量化计算优化](traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Loss 计算优化聚合](traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, operator_fusion
- 相关主题: [Matmul Scheduling 优化聚合](traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, gemm, matmul
- 相关主题: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](traj-matmul-split.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Misc 算子 Scheduling 优化聚合](traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, operator_fusion
- 相关主题: [MoE 稀疏 Weight Gradient 计算优化](traj-moe-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Reduction Compute 优化聚合](traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](traj-solve_tril-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [SSM Compute 优化聚合](traj-ssm-compute.md) — 主题相关：共享 910b2, 910b3, compute

<!-- okf:related:end -->
