---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: NsaCompressWithCache (NSA KV compression) on Ascend NPU
description: CANN ops-transformer 的 NsaCompressWithCache 是 Native-Sparse-Attention 推理阶段 KV 压缩算子：当某个 batch 的 token 数凑满一个 compressBlockSize 窗口时，将该 batch 后 compressBlockSize 个 token 用 weight 做加权求和压缩成一个 compresstoken，通过 slotmapping 写回 outputcache（README ...
tags: [nsa, attention, kv_cache, compression, elementwise, reduction, aiv_only, fp16, bf16, fp32_accumulation, tail_mask, contiguous_access, unified_buffer, ascend910b1, ascend910_93, kirin9030, kirinx90, nsa_compress_with_cache]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_compress_with_cache/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_compress_with_cache/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/nsa-compress-with-cache.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_compress_with_cache/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_compress_with_cache/)。

# NsaCompressWithCache（NSA KV 压缩）

CANN ops-transformer 的 `NsaCompressWithCache` 是 **Native-Sparse-Attention 推理阶段 KV 压缩算子**：当某个 batch 的 token 数凑满一个 `compressBlockSize` 窗口时，将该 batch 后 `compressBlockSize` 个 token 用 `weight` 做加权求和压缩成一个 `compress_token`，通过 `slot_mapping` 写回 `output_cache`（README L17-23）。它是 NSA 稀疏注意力流水线的上游步骤，压缩后的 KV 供下游稀疏 attention 使用。

## 1. 输入输出

### 输入（def.cpp L21-51）

| 输入 | 必选/可选 | 含义 | dtype | format |
|---|---|---|---|---|
| input | 必选 | 原始的 KV cache | FP16/BF16 | ND |
| weight | 必选 | 压缩权重 | FP16/BF16 | ND |
| slot_mapping | 必选 | 每个 batch 压缩结果存储位置索引 | INT32 | ND |
| output_cache | 必选（输入/输出同 tensor） | 输出的压缩后 cache | FP16/BF16 | ND |
| act_seq_len | 可选 | 实际序列长度 | INT64 | ND |
| block_table | 可选 | PageAttention 块表 | INT32 | ND |

### 输出（def.cpp L52-56）
- `output_cache`（REQUIRED，原地更新）

### 属性（def.cpp L57-61）
`layout`("TND")、`compress_block_size`(必选)、`compress_stride`(必选)、`act_seq_len_type`(必选)、`page_block_size`(必选)。

### 约束（README L93-105）
- compressBlockSize/stride 16 倍数，compressBlockSize≤64，compressBlockSize≥stride。
- headDim 16 倍数，headDim≤256；headNum≤64（>50 时偶数）。
- pageBlockSize=64 或 128；layout 仅 TND（属性值不生效）。
- 不支持 input/weight/output_cache 为空。

## 2. 核心公式

```
compressIdx = (seqLen - compressBlockSize) / stride
outputCache[slot_mapping[i]] = Σ(input[compressIdx*stride : compressIdx*stride+compressBlockSize] × weight[:])
```
（README L28-29）。weight 与 input 按第三维 broadcast。

## 3. 计算流程

| 阶段 | 操作 | 来源 |
|---|---|---|
| 1. InitBase | 读 tiling data，算每核 batch/head 范围，分配 ping-pong buffer | kernel L38-64 |
| 2. InitActSeqLen | 遍历 batch，检查 `(seqLen-compressBlockSize)%stride==0` | kernel L66-78 |
| 3. InitWeight | weight 搬入 UB，Cast to FP32 | kernel L80-91 |
| 4. Compress | 滑动窗口内 token 加权求和 → scatter store | Process |

## 4. tiling 与分核（详见 `artifacts/nsa-compress-with-cache/tiling_dispatch_tables.md`）

- 分核：按 head×batch 切分（`coreStartBatchIdxList`/`coreStartHeadIdxList`）。
- 核内：`headsNumPerCore`、`tokenNumPerTile`/`tokenRepeatCount`、`bufferNum` ping-pong。
- 架构：A2/A3（FP16/BF16）+ KirinX90/Kirin9030（仅 FP16）。

## 5. 架构支持（README L7-14；def.cpp L62-67）

| 产品 | 架构 | 支持 | dtype |
|---|---|---|---|
| Atlas A2/A3 训练/推理 | Ascend910B1/910_93 | √ | FP16/BF16 |
| Kirin X90 处理器系列 | KirinX90 | √ | FP16 |
| Kirin 9030 处理器系列 | Kirin9030 | √ | FP16 |

不支持 950/310P/训练系列。

## 6. Agent actions

1. 识别为 NSA KV 压缩（推理阶段滑动窗口加权求和 + scatter 写回）。
2. 按 batch 遍历，检查 `(seqLen-compressBlockSize)%stride==0` 触发压缩。
3. 滑动窗口 offset 计算：`start = compressIdx * stride`，长度 `compressBlockSize`。
4. weight broadcast 乘后 `tl.sum(axis=0)` 归约。
5. slot_mapping scatter 写回 output_cache。
6. PA 场景需通过 block_table 查物理 page 偏移。
7. `compressBlockSize` 和 `stride` 为 16 倍数，`compressBlockSize≤64`。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、weight 初始化方式（`weightInitFlag`）未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 `ops-transformer-nsa-compress-with-cache`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Legacy 缓存读写协议（triton-knowledge-retrieval）](../reference/SPEC-Retrieve-legacy.md) — 同属优化知识，主题相关
- 实践案例: [pad slot 单次检查提前退出，跳过无效填充序列](../runbooks/compute-pipeline/pad-slot-early-exit.md) — 主题相关：共享 kv_cache
- 实践案例: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 实践案例: [Add算子块大小动态调整优化经验](../runbooks/memory-sram/exp_elem_add_001_20260526.md) — 主题相关：共享 elementwise
- 实践案例: [带 alpha 缩放的逐元素 add 访存与块大小优化](../runbooks/memory-sram/exp_elemwise_add_001_20260526.md) — 主题相关：共享 elementwise
- 实践案例: [Memory Access Optimization: paged kv cache](../runbooks/memory-sram/exp_mla_paged_attn_001.md) — 主题相关：共享 attention
- 实践案例: [Memory Access Optimization: paged kv cache with gqa head tiling](../runbooks/memory-sram/exp_paged_gqa_001.md) — 主题相关：共享 attention
- 实践案例: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](../runbooks/memory-sram/parallel-axis-align.md) — 主题相关：共享 contiguous_access
- 相关主题: [Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡](add-broadcast-alpha.md) — 主题相关：共享 aiv_only, bf16, elementwise, fp16
- 相关主题: [ApplyRotaryPosEmb（RoPE 旋转位置编码 q/k 双路融合）Ascend C → Triton 语义转化卡](apply-rotary-pos-emb.md) — 主题相关：共享 aiv_only, bf16, elementwise, fp16
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, tail_mask, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, tail_mask, unified_buffer
- 相关主题: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](chunk-gated-delta-rule-fwd-h.md) — 主题相关：共享 bf16, fp16
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 aiv_only, ascend910b1, bf16, contiguous_access, elementwise, fp32_accumulation, tail_mask
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, bf16, elementwise, fp32_accumulation, tail_mask
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, tail_mask, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, tail_mask, unified_buffer
- 相关主题: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](fused-bias-leaky-relu.md) — 主题相关：共享 aiv_only, elementwise, fp16
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, tail_mask, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, kv_cache, tail_mask, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, kv_cache, tail_mask, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, kv_cache, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, contiguous_access, fp32_accumulation, kv_cache, tail_mask, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, kv_cache, tail_mask, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, elementwise, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, elementwise, fp16, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, nsa, tail_mask, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — NSA 压缩注意力带 KV cache 的变体
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, elementwise, fp32_accumulation, nsa, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, fp32_accumulation, nsa, tail_mask, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, kv_cache, tail_mask, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, tail_mask, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, attention, bf16, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, attention, bf16, elementwise, fp16, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, bf16, contiguous_access, fp32_accumulation, unified_buffer
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, kv_cache, tail_mask, unified_buffer

<!-- okf:related:end -->
