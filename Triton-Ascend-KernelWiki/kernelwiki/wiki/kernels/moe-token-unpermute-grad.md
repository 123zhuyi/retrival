---
id: kernel-moe-token-unpermute-grad
title: MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU
type: kernel
tags: [moe, token_unpermute, grad, backward, scatter, reduce_sum, aiv_only, fp16, bf16, fp32]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [moe]
operators: [moe_token_unpermute_grad]
symptoms: [precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-token-unpermute-grad
related: [kernel-moe-token-unpermute, kernel-moe-finalize-routing]
sources: [ops-transformer-moe-token-unpermute-grad]
performance_claims: []
---

# MoeTokenUnpermuteGrad（MoE token unpermute 反向）

CANN ops-transformer 的 `MoeTokenUnpermuteGrad` 是 [[kernel-moe-token-unpermute]] 的**反向传播**：计算 `permutedTokensGrad`（scatter 加权）和 `probsGrad`（topK 求和）（README L16）。**AIV-only**。

## 1. 输入输出

### 输入（def.cpp L33-60）
- permuted_tokens（REQUIRED，FP16/BF16/FP32）
- unpermuted_tokens_grad（REQUIRED，FP16/BF16/FP32）
- sorted_indices（REQUIRED，INT32）
- probs（OPTIONAL，FP16/BF16/FP32）

### 输出（def.cpp L61-72）
- permuted_tokens_grad（REQUIRED）
- probs_grad（REQUIRED）

### 属性（def.cpp L73-74）
`padded_mode`(false)/`restore_shape`({1,1})。

### 约束（README L118-129）
topK≤512；950 转调 finalize_routing_v2_grad。

## 2. 核心公式（README L19-45）

```
有 probs:
  probsGrad = Σ_{k}(unpermutedTokensGrad * permutedTokens[sortedIndices])   # topK 求和
  permutedTokensGrad[sortedIndices[k]] = unpermutedTokensGrad * probs[k]    # scatter 加权

无 probs (topK=1):
  permutedTokensGrad[sortedIndices[i]] = unpermutedOutputGrad[i]            # 纯 scatter
```

## 3. tiling key dispatch（4 key）

| key | padded_mode | haveProbs | 模板 |
|---|---|---|---|
| 0/10 | false/true | false | ProbNone |
| 1/11 | false/true | true | ProbNotNone |

## 4. 架构支持（README L7-12）

仅 Ascend910B1/910_93（A2/A3）。950 转调 finalize_routing_v2_grad。

## 5. 与 unpermute 配对

- 正向 `moe_token_unpermute`：gather + probs mul + reduce_sum
- 反向 `moe_token_unpermute_grad`：scatter 加权 + probsGrad reduce_sum

## 6. Agent actions

1. 识别为 MoE token unpermute 反向（scatter + 元素乘 + reduce_sum，AIV-only）。
2. 与 [[kernel-moe-token-unpermute]] 正向配对。
3. 有 probs 时输出双梯度（permutedTokensGrad + probsGrad），无 probs 时仅 permutedTokensGrad。
4. 950 转调 finalize_routing_v2_grad。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。