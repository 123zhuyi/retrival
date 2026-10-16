---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MoeTokenPermuteGrad (token permute backward) on Ascend NPU
description: CANN ops-transformer 中的 MoeTokenPermuteGrad 算子是 aclnnMoeTokenPermute 的反向传播算子（即 unpermute）。核心是 indexSelect + reshape + sum：对每个原始 token，将其被置换到 numTopk 个专家位置的梯度 gather 后沿 K 轴求和。A2/A3 直接实现，950 转调用 aclnnMoeInitRoutingV2Grad（框架代理）。
tags: [moe_token_permute_grad, moe, token_permute, backward, unpermute, index_select, reduce, fp32_accumulation, unified_buffer, vector_core, ascend910b1, ascend910b2, ascend910_93, ascend950pr, ascend950dt]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_permute_grad/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_permute_grad/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/moe-token-permute-grad.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_permute_grad/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/moe/moe_token_permute_grad/)。

# MoeTokenPermuteGrad (token permute backward)

CANN ops-transformer 中的 `MoeTokenPermuteGrad` 算子是 `aclnnMoeTokenPermute` 的反向传播算子（即 unpermute）。核心是 indexSelect + reshape + sum：对每个原始 token，将其被置换到 numTopk 个专家位置的梯度 gather 后沿 K 轴求和。A2/A3 直接实现，**950 转调用 `aclnnMoeInitRoutingV2Grad`**（框架代理）。

## 1. 输入输出

### 输入（按 `moe_token_permute_grad_def.cpp`）

| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| permutedOutputGrad | 必选 | 正向输出 permutedTokens 的梯度 [R*K, H] | BF16/FP16/FP32 |
| sortedIndices | 必选 | 排序索引 [R*K] | INT32 |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| out | 输出 token 的梯度 [R, H] | BF16/FP16/FP32 |

### 属性

| 属性 | 含义 | 说明 |
|------|------|------|
| num_topk | 被选中的专家个数 | REQUIRED，A2/A3 ≤ 512 |
| padded_mode | pad 模式开关 | OPTIONAL，默认 false |

### 约束

- A2/A3：numTopk <= 512
- A3：单卡通信量 [2MB, 100MB]
- 950：转调用 MoeInitRoutingV2Grad

## 2. 核心计算公式

```
inputGrad = permutedOutputGrad.indexSelect(0, sortedIndices)   # gather
inputGrad = inputGrad.reshape(-1, numTopk, hiddenSize)         # [R, K, H]
inputGrad = inputGrad.sum(dim=1)                                # 沿 K 轴 reduce -> [R, H]
```

## 3. 计算流程

```
for each token row:
    for k in 0..numTopk:
        idx = sortedIndices[row * numTopk + k]
        acc += permutedOutputGrad[idx]    # gather + 累加
    inputGrad[row] = acc
```

## 4. 模板与 dispatch

| TilingKey | dtype | 模板 |
|---|---|---|
| 0 | BF16 | `KernelMoeTokenUnpermute<bfloat16_t, int32_t, bfloat16_t, false>` |
| 2 | FP16 | `KernelMoeTokenUnpermute<half, int32_t, half, false>` |
| 4 | FP32 | `KernelMoeTokenUnpermute<float, int32_t, float, false>` |

> kernel 调用 `KernelMoeTokenUnpermute`（与 `moe_token_unpermute` 共享实现），probs 始终 false。

## 5. 架构与产品支持

| 产品 | 架构 | config（def.cpp） | kernel 入口 | 说明 |
|---|---|---|---|---|
| Atlas A2 训练/推理 | arch32 | `ascend910b` L46 | `moe_token_permute_grad.cpp` | 直接实现 |
| Atlas A3 训练/推理 | ascend910_93 | `ascend910_93` L47 | 同上 | 直接实现 |
| Ascend 950PR/950DT | arch35 | - | - | **转调用 `aclnnMoeInitRoutingV2Grad`** |

950 代理参数映射：
- permutedOutputGrad -> gradExpandedX
- sortedIndices -> expandedRowIdx
- numTopk -> topK
- paddedMode -> dropPadMode

## 6. Agent actions

1. 确认目标架构：A2/A3 直接实现；950 走 MoeInitRoutingV2Grad 代理。
2. 核心计算：indexSelect + reshape + sum -> `tl.load` + `tl.sum`。
3. Triton 实现参考 `artifacts/moe-token-permute-grad/moe-token-permute-grad-triton-skeleton.py`（pseudocode）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、950 代理正确性需结合源码/硬件进一步验证。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 `ops-transformer-moe-token-permute-grad`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Embedding / 索引拷贝算子优化](../reference/op-class-embedding.md) — 主题相关：共享 backward
- 实践案例: [RoPE（旋转位置编码）算子优化](../reference/op-class-rope.md) — 主题相关：共享 backward
- 实践案例: [跨 tile 归约必须显式两阶段 reduce-then-apply](../runbooks/algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 实践案例: [动态读取 Vector Core 数量，禁止硬编码](../runbooks/compute-pipeline/dynamic-vector-core-num.md) — 主题相关：共享 vector_core
- 实践案例: [Cross-kernel Optimization Transfer: token loop vs w loop decision](../runbooks/compute-pipeline/exp_causal_conv1d_update_006.md) — 同属优化知识，主题相关
- 实践案例: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](../runbooks/memory-sram/normalize-dim-to-last.md) — 同属优化知识，主题相关
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 ascend910b1, ascend950dt, ascend950pr, fp32_accumulation
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, backward, fp32_accumulation, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, backward, fp32_accumulation, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, backward, fp32_accumulation, moe, reduce, unified_buffer, vector_core
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, reduce, unified_buffer, vector_core
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer, vector_core
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, backward, fp32_accumulation, moe, reduce, unified_buffer, vector_core
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950dt, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, moe, unified_buffer, vector_core
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950dt, ascend950pr, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, moe, reduce, token_permute, unified_buffer, vector_core
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, moe, token_permute, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, fp32_accumulation, moe, token_permute, unified_buffer, vector_core
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 同一 MoeTokenPermute 算子的前向与反向实现
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, moe, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, moe, reduce, unified_buffer, vector_core
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, moe, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, backward, fp32_accumulation, reduce, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, fp32_accumulation, unified_buffer, vector_core

<!-- okf:related:end -->
