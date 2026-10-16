---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU
description: CANN ops-transformer 中的 QuantLightningIndexer 算子是推理场景下 SparseFlashAttention（SFA）的前处理算子，选出关键的稀疏 token 并对输入 query/key 进行量化实现存8算8。输出 sparseindices（INT32）供下游 SFA 算子使用。
tags: [sparse_flash_attention, sfa, lightning_indexer, quantized_attention, int8, fp8, page_attention, topk, inference, fp32_accumulation, tail_mask, unified_buffer, l0c_buffer, l1_buffer, l0c_to_ub, cube_vector_1_to_2, ascend910b1, ascend910_93, ascend950pr, attention, quant_lightning_indexer]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/quant_lightning_indexer/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/quant_lightning_indexer/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/quant-lightning-indexer.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/quant_lightning_indexer/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/quant_lightning_indexer/)。

# QuantLightningIndexer (SFA 量化稀疏索引选择)

CANN ops-transformer 中的 `QuantLightningIndexer` 算子是推理场景下 SparseFlashAttention（SFA）的前处理算子，选出关键的稀疏 token 并对输入 query/key 进行量化实现存8算8。输出 `sparse_indices`（INT32）供下游 SFA 算子使用。

支持 Atlas A2 推理系列（arch32, int8 输入）和 Ascend 950PR/950DT（arch35, fp8 输入）。支持 BSND、TND、PA_BSND（PageAttention）三种 layout。

## 1. 输入输出

### 输入（按 op_host/quant_lightning_indexer_def.cpp 顺序）

| 输入 | 必选/可选 | 含义 | dtype（arch32/arch35） |
|------|-----------|------|----------------------|
| query | 必选 | 量化查询 Q | int8 / fp8_e4m3fn, hifloat8 |
| key | 必选 | 量化键 K | int8 / fp8_e4m3fn, hifloat8 |
| weights | 必选 | 加权系数 W | FP16 / BF16 |
| query_dequant_scale | 必选 | Q 反量化系数 Scale_Q | FP16 / FLOAT |
| key_dequant_scale | 必选 | K 反量化系数 Scale_K^T | FP16 / FLOAT |
| actual_seq_lengths_query | 可选 | query 有效 token 数 | INT32 |
| actual_seq_lengths_key | 可选 | key 有效 token 数 | INT32 |
| block_table | 可选 | PA 块映射表 | INT32 |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| sparse_indices | 稀疏索引，shape [B,S1,N2,k] 或 [T,N2,k] | INT32 |

### 维度约定（README.md L32-33）

- D=128 固定（HEAD_DIM_LIMIT=128, tiling.h L75）
- N1（query head num）: arch32 支持 64, arch950 支持 24/64
- N2（key head num）: 仅支持 1
- G = N1/N2，G_SIZE_LIMIT=64, G_SIZE_LIMIT_950=24
- S1（query seq）、S2（key seq）、sparse_count [1,2048]
- block_size（PA）: 16 倍数，≤1024

### 属性

`query_quant_mode`(0 per-token-head)、`key_quant_mode`(0)、`layout_query`(BSND/TND)、`layout_key`(BSND/TND/PA_BSND)、`sparse_count`(2048)、`sparse_mode`(0/3)、`pre_tokens`(默认 INT64_MAX)、`next_tokens`(默认 INT64_MAX)

## 2. 核心公式（README.md L16-20）

```
out = Top_k{ W @ [ ReLU((Scale_Q ⊙ Scale_K^T) ⊙ (Q^Quant @ K^Quant^T)) ] }
```

| 步 | 公式 | 说明 |
|----|------|------|
| 1 | `Q^Quant @ K^Quant^T` | 量化 QK 相关性（int8/fp8 → FP32 累加）|
| 2 | `(Scale_Q ⊙ Scale_K^T) ⊙` | 反量化系数逐元素乘 |
| 3 | `ReLU()` | 过滤负相关信号 |
| 4 | `W @` | 加权聚合（跨 G 轴）|
| 5 | `Top_k(..., sparse_count)` | 选取前 k 个稀疏索引 |

## 3. 计算流程

| 阶段 | 入口 | 单元 | 操作 |
|------|------|------|------|
| Preload | QLIPreload::Init | AIC+AIV | 输入加载、反量化系数准备 |
| Cube | QKCompute | AIC | Q·K^T matmul（反量化 FP32） |
| Vector | IndexCompute | AIV | Scale_Q⊙Scale_K^T → ReLU → W@加权 |
| Topk | TopK | AIV | topk 索引选择 |
| WriteBack | WriteBack | AIV | sparse_indices 写回 GM |

## 4. Tiling 与 dispatch

- **模板**：单模板 `QLIPreload`，6 模板参数（DT_Q/DT_K/DT_OUT/PAGE_ATTENTION/Q_LAYOUT_T/K_LAYOUT_T）
- **arch32**：2 组合（int8 × PA/BSND/TND）
- **arch35**：6 组合（fp8_e4m3fn/hifloat8 × PA/BSND/TND）
- D=128 固定，S2 不切分（整块计算）
- `KERNEL_TYPE_MIX_AIC_1_2`（1 Cube:2 Vector）
- PageAttention：block_table 映射离散 KV 块

## 5. Agent actions

1. 确认架构（arch32 int8 vs arch35 fp8）与 layout（BSND/TND/PA_BSND）。
2. 加载反量化系数 Scale_Q/Scale_K^T，int8/fp8 输入在 FP32 计算。
3. 按 `Q·K^T → ReLU → W@ → topk` 编排计算。
4. sparse_mode=3 时应用因果掩码（rightDownCausal）。
5. PA_BSND 场景通过 block_table 索引离散 KV 块。
6. Triton 实现参考 `artifacts/quant-lightning-indexer/quant-lightning-indexer-triton-skeleton.py`（pseudocode）。

## 6. 复现性声明

本页结论基于源码静态分析（confidence: inferred），tiling 参数数值、UB 占用、PageAttention 运行时契约未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 `ops-transformer-quant-lightning-indexer`。

<!-- okf:related:start -->

# 相关

- 实践案例: [matmul core mapping 策略选择（列/行/对角）](../runbooks/compute-pipeline/core-mapping-strategy-selection.md) — 同属优化知识，主题相关
- 实践案例: [persistent kernel 模式（grid-stride 摊还启动开销）](../runbooks/compute-pipeline/persistent-kernel-pattern.md) — 主题相关：共享 inference
- 实践案例: [Tiling 预算与布局驱动的块大小选择](../runbooks/memory-sram/tiling-budget-and-layout-driven-selection.md) — 同属优化知识，主题相关
- 实践案例: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../runbooks/meta-skills/quant-fusion-int8-precision-diagnosis.md) — 主题相关：共享 int8
- 实践案例: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../runbooks/algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 attention
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, sparse_flash_attention, tail_mask, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, inference, l0c_buffer, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — lightning indexer 的量化变体
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, fp8, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, topk, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, topk, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, int8, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, int8, unified_buffer
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, int8, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, topk, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, topk, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, page_attention, sparse_flash_attention, tail_mask, unified_buffer

<!-- okf:related:end -->
