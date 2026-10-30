---
id: kernel-moe-token-unpermute-with-ep
title: MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU
type: kernel
tags: [moe, token_unpermute, ep, gather, reduce, aiv_only]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, register_file, vector_core]
architectures: [Ascend910B1, Ascend910_93, Kirin9030, KirinX90]
kernel_types: [moe]
operators: [moe_token_unpermute_with_ep]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-token-unpermute-with-ep
related: []
sources: [ops-transformer-moe-token-unpermute-with-ep]
performance_claims: []
---

# MoeTokenUnpermuteWithEp (MoE token unpermute with EP)

CANN ops-transformer 中的 `MoeTokenUnpermuteWithEp` 是 MoE token 反重排算子，根据 sortedIndices 从 permutedTokens 取数据（与 probs 相乘）并合并累加。纯 AIV-only Vector 算子（无 matmul），核心操作是 range 过滤 + gather(indexSelect) + 可选乘 probs + 跨 topK reduce sum。支持 A2 + Kirin（不支持 950）。

## 1. 输入输出

### 输入
| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| permuted_tokens | 必选 | 扩展并排序过的 tokens | BF16/FP16/FLOAT |
| sorted_indices | 必选 | 数据在 permutedTokens 中的位置 | INT32 |
| probs | 可选 | 专家概率 | BF16/FP16/FLOAT |

### 输出
| 输出 | 含义 | dtype |
|------|------|-------|
| unpermuted_tokens | 反重排输出 | BF16/FP16/FLOAT |

### 属性
`num_topk`(1)、`range`([0,0]，ep 切分范围)、`padded_mode`(false)、`restore_shape`（仅 nullptr）

## 2. 核心公式
1. `sortedIndices = sortedIndices[range[0] <= i < range[1]]` - range 过滤
2. `permutedTokens = permutedTokens.indexSelect(0, sortedIndices)` - gather
3. 若 probs 非 None：`permutedTokens_k *= probs_{i,j}` - 乘 probs
4. `out_i = Σ_{k=i*numTopk}^{(i+1)*numTopk-1} permutedTokens_k` - 跨 topK reduce sum

## 3. 模板 dispatch

4 tiling key：0（无 probs）/ 1（float probs）/ 2（half probs）/ 3（bf16 probs）。1 模板类 `KernelMoeTokenUnpermuteWithEp`，通过 `haveProbs` + probs dtype 区分。

## 4. 约束
- topK_num <= 512
- 不支持 paddedMode=True
- rangeOptional 为空时回退到 `MoeTokenUnpermute`
- ascend910b/910_93 + kirinX90/kirin9030（不支持 950）

## 5. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 [[ops-transformer-moe-token-unpermute-with-ep]]。