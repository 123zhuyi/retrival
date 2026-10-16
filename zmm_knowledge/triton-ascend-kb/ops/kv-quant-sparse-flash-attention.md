---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU
description: CANN ops-transformer 中的 KvQuantSparseFlashAttention 算子是在 sparseflashattention（SFA）基础上支持 Per-Token-Head-Tile-128 KV 量化的推理稀疏注意力算子。仅推理场景（A2/A3 推理 + 950），核心为 sparseindices 驱动的离散 KV 取数 + KV INT8/FP8 反量化 + MLA-absorb 模式（Q/K nope+rope 拼接 D=576/...
tags: [kv_quant_sparse_flash_attention, sparse_attention, flash_attention, quant, kv_cache, mla, inference, fp32_accumulation, sparse_kv_gather, sparse_attention_tiling, sparse_attention_pipeline, unified_buffer, l0c_buffer, cube_vector_1_to_2, ascend910b1, ascend910b2, ascend910_93, ascend950pr, ascend950dt, attention]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/kv_quant_sparse_flash_attention/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/kv_quant_sparse_flash_attention/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/kv-quant-sparse-flash-attention.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/kv_quant_sparse_flash_attention/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/kv_quant_sparse_flash_attention/)。

# KvQuantSparseFlashAttention (sparse inference with KV quant)

CANN ops-transformer 中的 `KvQuantSparseFlashAttention` 算子是在 `sparse_flash_attention`（SFA）基础上支持 **Per-Token-Head-Tile-128 KV 量化**的推理稀疏注意力算子。仅推理场景（A2/A3 推理 + 950），核心为 sparse_indices 驱动的离散 KV 取数 + KV INT8/FP8 反量化 + MLA-absorb 模式（Q/K nope+rope 拼接 D=576/656）。稀疏策略/离散访存优化见 `technique-sparse-kv-gather`、`technique-sparse-attention-tiling`、`technique-sparse-attention-pipeline`。

## 1. 输入输出

### 输入（按 README.md L29-84）

| 输入 | 必选/可选 | 含义 | dtype | 说明 |
|------|-----------|------|-------|------|
| query | 必选 | Q（nope+rope 拼接 D=576） | BF16/FP16 | BSND/TND |
| key | 必选 | K（nope+rope+dequant_scale 拼接 D=656） | INT8（arch32）/ FP8_E4M3FN/HIFLOAT8（arch35） | BSND/TND/PA_BSND |
| value | 必选 | V（同 key 量化） | 同 key | 同 key |
| sparse_indices | 必选 | 离散取数索引 [B, Q_S, KV_N, sparse_size] | INT32 | - |
| scale_value | 必选 | 缩放系数 1/√d | float | - |
| key_quant_mode | 必选 | 仅支持 2（per-tile） | int64 | - |
| value_quant_mode | 必选 | 仅支持 2（per-tile） | int64 | - |
| block_table | 可选 | PA block 映射表 | INT32 | - |
| actual_seq_lengths_query/kv | 可选 | 变长序列长度 | INT32 | - |
| sparse_block_size | 可选 | sparse 块大小（1-16，2 幂次） | int64 | 默认 1 |
| layout_query/kv | 可选 | 数据排布格式 | str | 默认 BSND |
| sparse_mode | 可选 | 3=rightDownCausal | int64 | 默认 3 |
| attention_mode | 可选 | 2=MLA-absorb | int64 | 默认 0 |
| quant_scale_repo_mode | 可选 | 1=combine mode | int64 | 默认 1 |
| tile_size | 可选 | per-tile 量化块大小 | int64 | 默认 128 |
| rope_head_dim | 可选 | MLA rope head dim | int64 | 默认 64 |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| attention_out | 注意力输出 | BF16/FP16 |

### 约束

- D=576（nope 512 + rope 64）；K D=656（nope 512 + rope 128 + dequant_scale 16）
- KV_N 仅支持 1
- sparse_block_size 整除 block_size
- 非 PA 场景 layout_query 与 layout_kv 一致

## 2. 核心计算公式

```
Attention = softmax((Q · Dequant(K^INT8, Scale_K)^T) / √d) · Dequant(V^INT8, Scale_V)
```

## 3. 计算流程

### 3.1 离散 KV 取数（sparse_indices 驱动）

```
sparse_indices[B, Q_S, KV_N, sparse_size] → 取 sparse_size 个 block 索引
→ 计算物理 KV 偏移 → 间接 load KV（INT8/FP8）
```

### 3.2 KV 反量化（Per-Token-Head-Tile-128）

量化参数和数据混合存放（quant_scale_repo_mode=1）：
- K D 维 = nope(512) + rope(128) + dequant_scale(4×4) = 656
- 反量化：`K_fp32 = cast(K_int8/fp8, fp32) * dequant_scale[tile_id]`
- per-tile: tile_size=128，每个 tile 共享一个 scale

### 3.3 MLA-absorb 模式（attention_mode=2）

Q/K 沿 D 维拼接 nope+rope（K 额外含 dequant_scale），分段 dot：
```
s = q_nope · k_nope^T + q_rope · k_rope^T
```

### 3.4 在线 FlashSoftmax + sparse mask

sparse_mode=3 rightDownCausal mask 后，FlashSoftmax 在线刷新，FP32 累加。

## 4. 模板与 dispatch

| 参数 | 取值 | 说明 |
|------|------|------|
| FLASH_DECODE | 0/1 | FlashDecode 使能 |
| LAYOUT_T | 0:BSND / 1:TND | query layout |
| KV_LAYOUT_T | 0:BSND / 1:TND / 2:PA_BSND | KV layout |
| TEMPLATE_MODE | 0:C_TEMPLATE / 1:V_TEMPLATE | Cube/Vector 模板 |

5 种合法组合（`template_tiling_key.h` L42-55）。

## 5. 架构与产品支持

| 产品 | 架构 | config（def.cpp） | kernel 入口 | KV dtype |
|------|------|-------------------|-------------|----------|
| Atlas A2 推理 | arch32 | `ascend910b` L92 | `kv_quant_sparse_flash_attention.cpp` | INT8 |
| Atlas A3 推理 | ascend910_93 | `ascend910_93` L93 | 同上 | INT8 |
| Ascend 950PR/950DT | arch35 | `ascend950` L151 | 同上（arch35 分支） | FP8_E4M3FN/HIFLOAT8 |

> 仅推理场景，无训练，无 310P。

## 6. Agent actions

1. 确认目标架构（arch32 INT8 / arch35 FP8）与 layout（BSND/TND/PA_BSND）。
2. sparse_indices 驱动离散取数：`tl.load(indices)` → 间接 `tl.load(KV)`。
3. KV per-tile-128 反量化：`cast(int8/fp8, fp32) * dequant_scale[tile_id]`。
4. MLA-absorb：Q/K 沿 D 拼接，分段 dot（nope + rope）。
5. 参考 `technique-sparse-kv-gather` 了解搬运聚合优化，`technique-sparse-attention-tiling` 了解 SFA tiling 策略，`technique-sparse-attention-pipeline` 了解流水编排。
6. Triton 实现参考 `artifacts/kv-quant-sparse-flash-attention/kv-quant-sparse-flash-attention-triton-skeleton.py`（pseudocode）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、per-tile 量化精度边界、sparse_indices 有效值约定、FP8_E4M3FN/HIFLOAT8 精度边界需结合源码/硬件进一步验证。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 `ops-transformer-kv-quant-sparse-flash-attention`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Fused Attention / Flash Attention 高性能算子](../reference/op-class-attention.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [Online Softmax / Flash Attention 单遍稳定计算](../runbooks/algorithm-numerics/online-softmax-flash-attention.md) — 主题相关：共享 flash_attention
- 实践案例: [Compute Optimization: sparse iteration strategy](../runbooks/compute-pipeline/exp_sparse_attn_topk_001.md) — 主题相关：共享 attention
- 实践案例: [Compute Optimization: sparse data access](../runbooks/compute-pipeline/exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 attention
- 实践案例: [pad slot 单次检查提前退出，跳过无效填充序列](../runbooks/compute-pipeline/pad-slot-early-exit.md) — 主题相关：共享 kv_cache
- 实践案例: [persistent kernel 模式（grid-stride 摊还启动开销）](../runbooks/compute-pipeline/persistent-kernel-pattern.md) — 主题相关：共享 inference
- 实践案例: [Memory Access Optimization: sparse gather optimization](../runbooks/memory-sram/exp_attn_sparse_bwd_001.md) — 主题相关：共享 attention
- 实践案例: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](../runbooks/memory-sram/exp_rmsnorm_fused_quant_001.md) — 同属优化知识，主题相关
- 实践案例: [Memory Access Optimization: sparse iteration logic](../runbooks/memory-sram/exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 attention
- 实践案例: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../runbooks/meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../runbooks/meta-skills/quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Flash Attention Causal Skeleton](../reference/skeleton-flash_attention_causal.md) — 同属优化知识，主题相关
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, sparse_attention, sparse_attention_tiling, sparse_kv_gather, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, flash_attention, fp32_accumulation, l0c_buffer, sparse_attention, sparse_attention_tiling, sparse_kv_gather, unified_buffer
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 ascend910b1, ascend950dt, ascend950pr, fp32_accumulation
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, flash_attention, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, flash_attention, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, attention, cube_vector_1_to_2, flash_attention, fp32_accumulation, kv_cache, l0c_buffer, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, attention, cube_vector_1_to_2, flash_attention, fp32_accumulation, kv_cache, l0c_buffer, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, sparse_attention, sparse_kv_gather, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, kv_cache, l0c_buffer, mla, quant, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, mla, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, kv_cache, l0c_buffer, mla, quant, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, quant, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, quant, unified_buffer
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, quant, unified_buffer
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, attention, fp32_accumulation, sparse_attention, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, kv_cache, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, flash_attention, fp32_accumulation, kv_cache, l0c_buffer, mla, quant, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, inference, l0c_buffer, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, sparse_attention, sparse_attention_tiling, unified_buffer
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — 主题相关：共享 ascend910b1
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — sparse FlashAttention 及其 KV 量化变体

<!-- okf:related:end -->
