---
id: kernel-moe-token-permute-with-ep-grad
title: MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU
type: kernel
tags: [moe, token_permute, ep, backward, gradient, gather, reduce, aiv_only]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, register_file, vector_core]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [moe]
operators: [moe_token_permute_with_ep_grad]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-token-permute-with-ep-grad
related: []
sources: [ops-transformer-moe-token-permute-with-ep-grad]
performance_claims: []
---

# MoeTokenPermuteWithEpGrad (MoE token permute with EP 反向)

CANN ops-transformer 中的 `MoeTokenPermuteWithEpGrad` 是 `MoeTokenPermuteWithEp` 的反向传播算子。纯 AIV-only Vector 算子（无 matmul），核心操作是 range 过滤 + gather(indexSelect) + reshape + 跨 topK reduce sum。仅支持 Atlas A2（不支持 950）。

## 1. 输入输出

### 输入
| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| permuted_tokens_output_grad | 必选 | 正向 permutedTokens 的梯度 | BF16/FP16/FLOAT |
| sorted_indices | 必选 | permute 映射关系 | INT32 |
| permuted_probs_output_grad | 可选 | probs 梯度（用于 probs_grad 输出）| BF16/FP16/FLOAT |

### 输出
| 输出 | 含义 | dtype |
|------|------|-------|
| token_grad | 输入 token 的梯度 | BF16/FP16/FLOAT |
| probs_grad | 输入 probs 的梯度 | BF16/FP16/FLOAT |

### 属性
`num_topk`（必需）、`range`（[0,0] 默认，ep 切分有效范围）、`padded_mode`（false，仅支持 false）

## 2. 核心公式
1. `sortedIndices = sortedIndices[range[0] <= i < range[1]]` - range 过滤
2. `tokenGradOut = permutedTokensOutputGrad.indexSelect(0, sortedIndices)` - gather
3. `tokenGradOut = tokenGradOut.reshape(-1, topK, hiddenSize)` - reshape
4. `tokenGradOut = tokenGradOut.sum(dim=1)` - 跨 topK reduce sum

## 3. 模板 dispatch

`KERNEL_TYPE_AIV_ONLY`。2 tiling key：0（无 probs，仅 token_grad）/ 1（有 probs，token_grad + probs_grad）。1 模板类 `KernelMoeTokenUnpermuteWithEp`，通过 `haveProbs` 参数区分。

## 4. 约束
- top_k <= 512
- 不支持 paddedMode=True
- rangeOptional 为空时回退到 `MoeTokenPermuteGrad`
- 仅 ascend910b/910_93（不支持 950）

## 5. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 [[ops-transformer-moe-token-permute-with-ep-grad]]。