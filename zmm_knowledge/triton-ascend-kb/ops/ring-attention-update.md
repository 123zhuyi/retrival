---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU
description: CANN ops-transformer 的 RingAttentionUpdate 是 Ring Attention 的输出合并算子：将两次 FlashAttention 的输出根据各自的 softmax max/sum 做在线 rescale 合并，生成统一的 attention 输出（README L14-15）。本算子只做 rescale 合并，不做 QK/PV，其逻辑与 FlashAttention 内部的 FlashSoftmax rescale 步骤完全一...
tags: [ring_attention, attention, softmax, online_softmax, elementwise, aiv_only, fp16, bf16, fp32, fp32_accumulation, unified_buffer, ascend910b1, ascend910_93, ascend950pr, kirin9030, kirinx90, ring_attention_update]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/ring_attention_update/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/ring_attention_update/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/ring-attention-update.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/ring_attention_update/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/ring_attention_update/)。

# RingAttentionUpdate（Ring Attention 输出 rescale 合并）

CANN ops-transformer 的 `RingAttentionUpdate` 是 **Ring Attention 的输出合并算子**：将两次 FlashAttention 的输出根据各自的 softmax max/sum 做在线 rescale 合并，生成统一的 attention 输出（README L14-15）。**本算子只做 rescale 合并，不做 QK/PV**，其逻辑与 FlashAttention 内部的 FlashSoftmax rescale 步骤完全一致（`exp[i]=e^(max[i-1]-max[i])`），是将"上一步 vs 当前步"改为"第一次 vs 第二次 FA 输出"。

## 1. 输入输出

### 输入（def.cpp L22-63）

| 输入 | 必选/可选 | 含义 | dtype |
|---|---|---|---|
| prev_attn_out | 必选 | 第一次 FA 输出 | FLOAT/FP16/BF16 |
| prev_softmax_max | 必选 | 第一次 softmax max | FLOAT |
| prev_softmax_sum | 必选 | 第一次 softmax sum | FLOAT |
| cur_attn_out | 必选 | 第二次 FA 输出 | FLOAT/FP16/BF16 |
| cur_softmax_max | 必选 | 第二次 softmax max | FLOAT |
| cur_softmax_sum | 必选 | 第二次 softmax sum | FLOAT |
| actual_seq_qlen | 可选 | TND 时必填，前缀和 | INT64 |

### 输出（def.cpp L64-78）
- attn_out / softmax_max / softmax_sum

### 属性（def.cpp L79-80）
`input_layout`("SBH")、`input_softmax_layout`("")。

### 约束（README L127-131）
TND 时 D 为 64 倍数，actual_seq_qlen 必填。`(N*D)_align64 * (dtype_size*6+8) + (N*8)_align64 * 56 ≤ 192*1024`。

## 2. 核心公式（README L18-28）

```
new_max = max(prev_max, cur_max)
alpha_prev = exp(prev_max - new_max);  alpha_cur = exp(cur_max - new_max)
new_sum = prev_sum * alpha_prev + alpha_cur
attn_out = prev_out * alpha_prev / new_sum + cur_out * alpha_cur / new_sum
```

## 3. tiling key dispatch

| key | dtype | layout | kernel |
|---|---|---|---|
| 0 | FP16 | SBH | arch32 |
| 1 | BF16 | SBH | arch32 |
| 2 | FLOAT | SBH | arch32 |
| 10 | FP16 | TND | arch32 |
| 11 | BF16 | TND | arch32 |
| 12 | FLOAT | TND | arch32 |
| 100 | - | SBH | arch35 |
| 110 | - | SoftmaxTND | arch35 |
| 120 | - | TND | arch35 |

## 4. 架构支持

A2/A3(arch32) + Ascend950(arch35) + KirinX90/Kirin9030(仅 FP16/FLOAT)。

## 5. Agent actions

1. 识别为 **Ring Attention 输出 rescale 合并**（纯 elementwise，AIV-only）。
2. 核心逻辑与 FlashAttention 的 FlashSoftmax rescale 一致：`alpha=exp(max_prev-max_new)`，`out = prev*alpha_prev/sum + cur*alpha_cur/sum`。
3. SBH layout = 标准 3D，TND layout = 前缀和寻址。
4. Triton 中 `tl.maximum` + `tl.exp` + 标量乘除即可。

## 6. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Softmax / LogSoftmax 算子优化](../reference/op-class-softmax.md) — 主题相关：共享 online_softmax, softmax
- 实践案例: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 softmax
- 实践案例: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](../runbooks/algorithm-numerics/exp_stability_001.md) — 同属优化知识，主题相关
- 实践案例: [Online Softmax / Flash Attention 单遍稳定计算](../runbooks/algorithm-numerics/online-softmax-flash-attention.md) — 主题相关：共享 online_softmax
- 实践案例: [按输出元素数（而非输入）分配 grid，且钳制到 num_cores](../runbooks/compute-pipeline/grid-by-output-count.md) — 同属优化知识，主题相关
- 实践案例: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 实践案例: [Memory Access Optimization: fused state update](../runbooks/memory-sram/exp_causal_conv1d_update_001.md) — 同属优化知识，主题相关
- 实践案例: [Add算子块大小动态调整优化经验](../runbooks/memory-sram/exp_elem_add_001_20260526.md) — 主题相关：共享 elementwise
- 实践案例: [带 alpha 缩放的逐元素 add 访存与块大小优化](../runbooks/memory-sram/exp_elemwise_add_001_20260526.md) — 主题相关：共享 elementwise
- 实践案例: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](../runbooks/memory-sram/exp_fused_attention_002.md) — 主题相关：共享 attention
- 实践案例: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../runbooks/meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 attention
- 相关主题: [Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡](add-broadcast-alpha.md) — 主题相关：共享 aiv_only, bf16, elementwise, fp16, fp32
- 相关主题: [ApplyRotaryPosEmb（RoPE 旋转位置编码 q/k 双路融合）Ascend C → Triton 语义转化卡](apply-rotary-pos-emb.md) — 主题相关：共享 aiv_only, bf16, elementwise, fp16, fp32
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, online_softmax, unified_buffer
- 相关主题: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](chunk-gated-delta-rule-fwd-h.md) — 主题相关：共享 bf16, fp16
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 aiv_only, ascend910b1, ascend950pr, bf16, elementwise, fp32_accumulation
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, bf16, elementwise, fp32_accumulation
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, fp32_accumulation, online_softmax, softmax, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, fp32_accumulation, online_softmax, softmax, unified_buffer
- 相关主题: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](fused-bias-leaky-relu.md) — 主题相关：共享 aiv_only, elementwise, fp16, fp32
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, softmax, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, fp32_accumulation, online_softmax, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, fp32_accumulation, online_softmax, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, elementwise, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, elementwise, fp16, fp32, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, softmax, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, bf16, fp16, fp32, fp32_accumulation, softmax, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, bf16, fp16, fp32, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp16, fp32, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, attention, bf16, elementwise, fp16, fp32_accumulation, kirin9030, kirinx90, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, fp32_accumulation, online_softmax, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, fp32_accumulation, online_softmax, softmax, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, online_softmax, softmax, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, attention, bf16, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, online_softmax, softmax, unified_buffer

<!-- okf:related:end -->
