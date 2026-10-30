---
id: kernel-moe-token-unpermute
title: MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU
type: kernel
tags: [moe, token_unpermute, gather, reduce_sum, aiv_only, fp16, bf16, fp32]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [moe]
operators: [moe_token_unpermute]
symptoms: [precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-token-unpermute
related: [kernel-moe-token-permute, kernel-moe-finalize-routing]
sources: [ops-transformer-moe-token-unpermute]
performance_claims: []
---

# MoeTokenUnpermute（MoE token 反向重排）

CANN ops-transformer 的 `MoeTokenUnpermute` 是 [[kernel-moe-token-permute]] 的**逆操作**：按 sortedIndices 从 permutedTokens gather，若有 probs 相乘，再按 topK 累加求和还原回原始 token 顺序（README L18）。**AIV-only**（gather + probs mul + reduce_sum，无 Cube）。

## 1. 输入输出

### 输入（def.cpp L23-58）
- permuted_tokens（REQUIRED，FP16/BF16/FP32）
- sorted_indices（REQUIRED，INT32）
- probs（OPTIONAL，FP16/BF16/FP32）

### 输出（def.cpp L63-73）
- unpermuted_tokens（REQUIRED，同 permuted_tokens dtype）

### 属性（def.cpp L60-61）
`padded_mode`(false)/`restore_shape`(OPTIONAL ListInt)。

### 约束（README L106-119）
topK≤512；950 转调 finalize_routing_v2；310P 仅 FP16/FP32，hiddensize 128 倍数≤10240。

## 2. 核心公式（README L21-47）

```
有 probs:
  T[k] = permutedTokens[sortedIndices[k]]       # gather
  T[k] = T[k] * probs[k]                          # probs 乘
  O[i] = Σ_{k=i*topK}^{(i+1)*topK-1} T[k]        # topK 累加还原

无 probs (topK=1):
  O[i] = permutedTokens[sortedIndices[i]]         # 纯 gather
```

## 3. tiling key dispatch（~16 key）

dtype(BF16/FP16/FP32) × haveProbs(false/true) × mix mode（probs dtype ≠ token dtype，key 11/13/17/21/25/27）。

## 4. 架构支持（README L7-14）

A2/A3(BF16/FP16/FP32) + 310P/Kirin(仅 FP16/FP32)。950 转调 finalize_routing_v2。

## 5. 与 token_permute 配对

- `moe_token_permute`：tokens -> permuteTokens（按 indices 排序展开）
- `moe_token_unpermute`：permutedTokens -> unpermutedTokens（按 sortedIndices gather + probs 加权 + topK 累加还原）

## 6. Agent actions

1. 识别为 MoE token 反向重排（gather + probs mul + reduce_sum，AIV-only）。
2. 与 [[kernel-moe-token-permute]] 配对：permute 排序展开，unpermute gather+累加还原。
3. mix mode 支持 probs dtype 与 token dtype 不同（如 FP16 token + FP32 probs）。
4. 950 转调 finalize_routing_v2。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。