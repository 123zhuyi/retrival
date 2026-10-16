---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU
description: CANN ops-transformer 中的 FusedFloydAttention 算子使用 FloydAttention 算法实现多维自注意力计算，支持 2 组 KV head group（K1/V1 + K2/V2）。仅支持 Atlas A2 训练系列（arch32 / ascend910b / ascend91093），不支持 ascend950。
tags: [floyd_attention, attention, multi_dim, dual_kv, softmax, matmul, cube_vector_pipe, fp32_accumulation, tail_mask, unified_buffer, l0c_buffer, l1_buffer, l0c_to_ub, cube_vector_1_to_2, ascend910b1, ascend910_93, fused_floyd_attention]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/fused_floyd_attention/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/fused_floyd_attention/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/fused-floyd-attention.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/fused_floyd_attention/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/fused_floyd_attention/)。

# FusedFloydAttention (Floyd 多维注意力)

CANN ops-transformer 中的 `FusedFloydAttention` 算子使用 FloydAttention 算法实现多维自注意力计算，支持 2 组 KV head group（K1/V1 + K2/V2）。仅支持 Atlas A2 训练系列（arch32 / ascend910b / ascend910_93），不支持 ascend950。

## 1. 输入输出

### 输入
| 输入 | 必选 | 含义 | dtype |
|------|------|------|-------|
| query | 是 | 查询 Q | FP16/BF16/FLOAT |
| key1 | 是 | 键 K1 | FP16/BF16/FLOAT |
| value1 | 是 | 值 V1 | FP16/BF16/FLOAT |
| key2 | 是 | 键 K2 | FP16/BF16/FLOAT |
| value2 | 是 | 值 V2 | FP16/BF16/FLOAT |
| atten_mask | 可选 | 注意力掩码 | BOOL/UINT8 |

### 输出
| 输出 | 含义 | dtype |
|------|------|-------|
| softmax_max | Softmax 行最大值 | FLOAT |
| softmax_sum | Softmax 行求和 | FLOAT |
| attention_out | 注意力输出 | FP16/BF16/FLOAT |

### 约束（README.md L112-124）
B:1~2K, H:1~256, N:16~1M(N%16==0), M:128~1M(M%128==0), K:128~1M(K%128==0), D:32/64/128

## 2. 核心公式

```
weights = Softmax(attenMask + scale * (Q@K1^T + Q@K2^T))
attention_out = weights@V1 + weights@V2
```

4 次 matmul: Q·K1^T + Q·K2^T + P·V1 + P·V2

## 3. 计算流程

| 阶段 | 入口 | 单元 | 操作 |
|------|------|------|------|
| Cube1 | IterateBmm1 | AIC | Q·K1^T |
| Cube1k1 | IterateBmm1K1 | AIC | Q·K2^T |
| Vector1 | ProcessVec1 | AIV | QK1+QK2 → scale → mask → Softmax |
| Cube2 | IterateBmm2 | AIC | P·V1 |
| Cube2v2 | IterateBmm2V2 | AIC | P·V2 |
| Vector2 | ProcessVec2 | AIV | P·V1+P·V2 → 写回 |

## 4. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 `ops-transformer-fused-floyd-attention`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Fused Attention / Flash Attention 高性能算子](../reference/op-class-attention.md) — 主题相关：共享 attention, softmax
- 实践案例: [MatMul / GEMM 高性能算子](../reference/op-class-matmul.md) — 主题相关：共享 matmul
- 实践案例: [Softmax / LogSoftmax 算子优化](../reference/op-class-softmax.md) — 主题相关：共享 softmax
- 实践案例: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 softmax
- 实践案例: [Online Softmax / Flash Attention 单遍稳定计算](../runbooks/algorithm-numerics/online-softmax-flash-attention.md) — 同属优化知识，主题相关
- 实践案例: [matmul core mapping 策略选择（列/行/对角）](../runbooks/compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 实践案例: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../runbooks/compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 实践案例: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 同属优化知识，主题相关
- 实践案例: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../runbooks/memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 实践案例: [多维输入必须在 host 侧扁平化为 2D](../runbooks/memory-sram/host-flatten-to-2d.md) — 主题相关：共享 multi_dim
- 实践案例: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](../runbooks/memory-sram/normalize-dim-to-last.md) — 同属优化知识，主题相关
- 实践案例: [Cube-Vector 场景行宽必须 512B 对齐](../runbooks/memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 实践案例: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../runbooks/meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 attention
- 实践案例: [tl.dot vs torch.matmul 精度差异诊断](../runbooks/meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [按 (维度, 模式, 采样比例) 多策略分派，禁止单 kernel 统所有路径](../runbooks/meta-skills/multi-strategy-dispatch.md) — 同属优化知识，主题相关
- 实践案例: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Rmsnorm Fused Skeleton](../reference/skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [大序列 Attention 的分块计算与 Online Softmax 数值稳定](../runbooks/algorithm-numerics/traj-attention-memory_numerical.md) — 主题相关：共享 attention
- 相关主题: [Matmul Compute 优化聚合](../runbooks/compute-pipeline/traj-matmul-compute.md) — 主题相关：共享 matmul
- 相关主题: [Matmul Scheduling 优化聚合](../runbooks/compute-pipeline/traj-matmul-scheduling.md) — 主题相关：共享 matmul
- 相关主题: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](../runbooks/compute-pipeline/traj-matmul-split.md) — 主题相关：共享 matmul
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](../runbooks/compute-pipeline/traj-softmax-compute.md) — 主题相关：共享 softmax
- 相关主题: [Softmax Backward 内存访问优化聚合](../runbooks/memory-sram/traj-softmax-memory_access.md) — 主题相关：共享 softmax
- 相关主题: [Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐](../runbooks/meta-skills/traj-attention-workflow_opt.md) — 主题相关：共享 attention, softmax
- 相关主题: [Matmul 硬件适配优化聚合](../runbooks/portability/traj-matmul-hardware_adaptation.md) — 主题相关：共享 matmul
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, softmax, tail_mask, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, softmax, tail_mask, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 同一 FusedFloydAttention 算子的前向与反向实现
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, matmul, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, softmax, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, softmax, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, tail_mask, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, softmax, tail_mask, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, softmax, tail_mask, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, softmax, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, softmax, tail_mask, unified_buffer

<!-- okf:related:end -->
