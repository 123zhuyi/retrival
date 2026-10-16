---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU
description: CANN ops-transformer 中的 MoeGatingTopKSoftmaxV2 是 MoE gating 的 topk + softmax 算子，按 renorm 标记区分 softmax 与 topk 的先后顺序。纯 AIV-only Vector 算子（无 matmul）。
tags: [moe, gating, topk, softmax, aiv_only, fp32_accumulation, unified_buffer, register_file, vector_core, ascend910b1, ascend910_93, ascend950pr, moe_gating_top_k_softmax_v2]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_gating_top_k_softmax_v2/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_gating_top_k_softmax_v2/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/moe-gating-top-k-softmax-v2.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_gating_top_k_softmax_v2/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_gating_top_k_softmax_v2/)。

# MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax)

CANN ops-transformer 中的 `MoeGatingTopKSoftmaxV2` 是 MoE gating 的 topk + softmax 算子，按 renorm 标记区分 softmax 与 topk 的先后顺序。纯 AIV-only Vector 算子（无 matmul）。

## 1. 输入输出

### 输入
| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| x | 必选 | gating 输入 | FP16/FLOAT/BF16 |
| finished | 可选 | 行完成标记（True 则 expert_idx=num_expert） | BOOL |

### 输出
| 输出 | 含义 | dtype |
|------|------|-------|
| y | softmax 的 topk 结果 | FP16/FLOAT/BF16 |
| expert_idx | topk 的 indices（专家序号） | INT32 |
| softmax_result | softmax 结果（可选） | FLOAT |

### 属性
`k`（必需）、`renorm`(0)、`output_softmax_result_flag`(false)

## 2. 核心公式
- **renorm=0**：`softmaxResult = softmax(x, axis=-1)`；`yOut, expertIdxOut = topk(softmaxResult, k)`
- **renorm=1**：`topkOut, expertIdxOut = topk(x, k)`；`yOut = softmax(topkOut, axis=-1)`
- finished=True 时 expertIdxOut = num_expert

## 3. 模板 dispatch

`KERNEL_TYPE_MIX_AIV_1_0`（AIV-only）。~36 tiling key，4 模板策略：
- **EKFullLoad**（101xxx）：expert+K 全加载
- **KFullLoad**（102xxx，renorm=0）：K 全加载
- **KRenorm**（102xxx，renorm=1）：K renorm
- **Perf**（103xxx）：高性能，按 colRange 分 3 档（<8 / 8-64 / >64）

## 4. 约束
- k <= 1024
- renorm 只支持 0/1
- x 和 finished 每一维 <= INT32_MAX

## 5. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 `ops-transformer-moe-gating-top-k-softmax-v2`。

<!-- okf:related:start -->

# 相关

- 实践案例: [归一化算子优化](../reference/op-class-normalization.md) — 主题相关：共享 softmax
- 实践案例: [Softmax / LogSoftmax 算子优化](../reference/op-class-softmax.md) — 主题相关：共享 softmax
- 实践案例: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 softmax
- 实践案例: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](../runbooks/algorithm-numerics/exp_stability_001.md) — 同属优化知识，主题相关
- 实践案例: [Online Softmax / Flash Attention 单遍稳定计算](../runbooks/algorithm-numerics/online-softmax-flash-attention.md) — 同属优化知识，主题相关
- 实践案例: [动态读取 Vector Core 数量，禁止硬编码](../runbooks/compute-pipeline/dynamic-vector-core-num.md) — 主题相关：共享 vector_core
- 实践案例: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](../runbooks/memory-sram/exp_fused_attention_002.md) — 同属优化知识，主题相关
- 实践案例: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../runbooks/meta-skills/attention-fusion-failure-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](../runbooks/portability/exp_sparse_flash_attn_prefill_001.md) — 同属优化知识，主题相关
- 相关主题: [ApplyRotaryPosEmb（RoPE 旋转位置编码 q/k 双路融合）Ascend C → Triton 语义转化卡](apply-rotary-pos-emb.md) — 主题相关：共享 aiv_only
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, softmax, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, softmax, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, softmax, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, topk, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — gating top-k softmax 版本迭代
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer, vector_core
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, register_file, unified_buffer, vector_core
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, topk, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, topk, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, softmax, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, topk, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, softmax, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, softmax, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, softmax, unified_buffer

<!-- okf:related:end -->
