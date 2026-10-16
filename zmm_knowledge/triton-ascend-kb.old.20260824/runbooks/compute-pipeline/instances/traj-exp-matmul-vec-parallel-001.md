---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: compute-pipeline
dimension: "2.1"
title: "exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion"
description: 使用 dl.parallel 在 M 维拆分 accumulator sub-block，配合 grouped ordering 提升 L2 复用，并在 FP32 累加器上融合 leaky_relu_custom 激活。
tags: [matmul, gemm, compute_optimization, vector_parallelism, activation_fusion, autotune, grouped_ordering, l2_reuse, fp32_accumulate, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/matmul/exp_matmul_vec_parallel_001.yaml
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion

## 概述

本实例展示如何在 Ascend 910B 上使用 `dl.parallel` 对 matmul accumulator 做 M 维向量并行拆分，并直接在 FP32 累加器上融合 `leaky_relu_custom` 激活。核心收益来自：sub-block 级 Vector 并行提升指令吞吐、grouped ordering 改善 L2 复用、以及 activation fusion 减少一次 elementwise kernel 启动。

## 适用条件

| 维度 | 要求 |
|---|---|
| 算子类型 | Matrix Operations |
| 计算模式 | Matrix Multiplication |
| 数据类型 | fp16 / bf16 |
| 硬件 | 910b2 / 910b3 |
| shape | M ≥ 32, N ≥ 32, K ≥ 32 |
| tensor shape | nd（原生排布，无需预转置） |
| 优化级别 | advanced |

## 优化点

### 1. 向量并行策略（`dl.parallel`）

当 `VEC_PARALLEL != 1` 时，accumulator 沿 M 维拆分为 `SUB_BLK_M` sub-block，通过 `dl.parallel(0, 2, bind_sub_block=True)` 在 Vector 单元上并行处理，提升指令级并行度。

### 2. Grouped ordering 提升 L2 复用

`GROUP_SIZE_M = 1` 的 grouped PID 映射确保相邻 program 处理的输出块在 L2 中相邻，促进矩阵 A 的跨块复用。

### 3. Activation fusion

`leaky_relu_custom` 直接在 FP32 accumulator 上执行，cast 回 fp16 前完成，避免单独启动 element-wise kernel。

### 4. Autotune 集成

`@triton.autotune(configs=get_autotune_config(), key=["M", "N", "K"])` 在运行时搜索最优分块配置。

## 性能收益

**35%**（相对于未启用向量并行与 fusion 的 baseline）

## Kernel 代码

```python
@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    VEC_PARALLEL: tl.constexpr,
):
    GROUP_SIZE_M: tl.constexpr = 1
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs_base = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs_base = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    msk_m = offs_am < M
    msk_n = offs_bn < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_ptrs = a_ptrs_base + k * BLOCK_SIZE_K * stride_ak
        b_ptrs = b_ptrs_base + k * BLOCK_SIZE_K * stride_bk
        a = tl.load(
            a_ptrs,
            mask=msk_m[:, None] and (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=msk_n[None, :] and (offs_k[:, None] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)

    if VEC_PARALLEL == 1:
        if ACTIVATION == "leaky_relu_custom":
            accumulator = leaky_relu_custom(accumulator)
        c = accumulator.to(tl.float16)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        SUB_BLK_M: tl.constexpr = BLOCK_SIZE_M // 2
        for s in dl.parallel(0, 2, bind_sub_block=True):
            vec_sub_blk = dl.extract_slice(
                accumulator, (s * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_SIZE_N), (1, 1)
            )
            if ACTIVATION == "leaky_relu_custom":
                vec_sub_blk = leaky_relu_custom(vec_sub_blk)
            c_sub_blk = vec_sub_blk.to(tl.float16)
            offs_cm = pid_m * BLOCK_SIZE_M + s * SUB_BLK_M + tl.arange(0, SUB_BLK_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c_sub_blk, mask=c_mask)
```

## Note

The kernel utilizes hardware-specific dialect features (e.g., `dl.parallel`) indicative of Ascend NPU optimization. It supports generic memory layouts via stride arguments, making it versatile for different tensor formats without pre-transposition.

## 关联知识

- 开发指南: [MatMul / GEMM 高性能算子](../../../reference/op-class-matmul.md)
- 讲解接口: [matmul core mapping 策略选择（列/行/对角）](../core-mapping-strategy-selection.md)
- 聚合视图: [matmul Compute Optimization 实践聚合](../traj-matmul-compute.md)
- 对应 YAML: `raw/experiences/matmul/exp_matmul_vec_parallel_001.yaml`
- 对应 Kernel: `raw_experiences/extracted_kernels/072-matmul_kernel.py`

<!-- okf:related:start -->

# 相关

- 实践案例: [Triton API Quick Reference for Ascend Development](../../../reference/triton-api-cheatsheet.md) — 主题相关：共享 910b2, 910b3, autotune
- 对应指南: [Elementwise 算子优化](../../../reference/op-class-elementwise.md) — 主题相关：共享 910b2, 910b3
- 对应指南: [融合算子综合页](../../../reference/op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, gemm
- 对应指南: [MatMul / GEMM 高性能算子](../../../reference/op-class-matmul.md) — 主题相关：共享 910b2, 910b3, gemm, matmul
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../../ops/mla-preprocess.md) — 主题相关：共享 matmul
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../../ops/mla-prolog.md) — 主题相关：共享 matmul
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../../ops/moe-token-permute-with-ep.md) — 同属优化知识，主题相关
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../../ops/rain-fusion-attention.md) — 主题相关：共享 matmul
- 相关主题: [Gemm Basic Skeleton](../../../reference/skeleton-gemm_basic.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 fp32_accumulate
- 相关主题: [GEMM 操作数保持 low_dtype，fp32 累加](../../algorithm-numerics/gemm-low-dtype-fp32-accum.md) — 主题相关：共享 gemm
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](../../algorithm-numerics/traj-general_stability-numerical_stability.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [累加/原地修改算子 autotune 必须每次重置输出 buffer](../../autotuning/autotune-debug.md) — 主题相关：共享 autotune
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../../autotuning/autotune-num-warps-stages.md) — 主题相关：共享 autotune
- 相关主题: [扩展 Autotune 配置空间以覆盖多 Shape 最优调度](../../autotuning/traj-general_perf-tuning_config.md) — 主题相关：共享 910b2, 910b3, autotune
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](../../memory-sram/parallel-axis-align.md) — 同属优化知识，主题相关
- 相关主题: [Cube-Vector 场景行宽必须 512B 对齐](../../memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../../memory-sram/tiling-budget-and-layout-driven-selection.md) — 主题相关：共享 autotune
- 相关主题: [Activation Backward 内存访问优化聚合](../../memory-sram/traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../../meta-skills/attention-fusion-failure-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../../meta-skills/quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../../meta-skills/quant-matmul-autotune-strategy.md) — 主题相关：共享 autotune
- 相关主题: [Matmul 硬件适配优化聚合](../../portability/traj-matmul-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, matmul
- 相关主题: [dl.async_task 实现 Cube/Vector 异构并行](../async-task-cube-vector.md) — 同属优化知识，主题相关
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](../core-mapping-strategy-selection.md) — 主题相关：共享 l2_reuse, matmul
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 相关主题: [算子融合模式与约束决策](../operator-fusion-patterns.md) — 主题相关：共享 gemm
- 相关主题: [swizzle2d 块重排提升 L2 复用](../swizzle2d-block-reorder.md) — 主题相关：共享 l2_reuse
- 相关主题: [SwiGLU 计算优化：门控融合、展平布局与激活量化](../traj-activation-compute.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Activation Scheduling 优化聚合](../traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Matmul Compute 优化聚合](../traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, activation_fusion, gemm, matmul, vector_parallelism
- 相关主题: [Matmul Scheduling 优化聚合](../traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, gemm, matmul
- 相关主题: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](../traj-matmul-split.md) — 主题相关：共享 910b2, 910b3, matmul

<!-- okf:related:end -->
