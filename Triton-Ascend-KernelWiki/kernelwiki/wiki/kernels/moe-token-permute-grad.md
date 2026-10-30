---
id: kernel-moe-token-permute-grad
title: MoeTokenPermuteGrad (token permute backward) on Ascend NPU
type: kernel
tags: [moe_token_permute_grad, moe, token_permute, backward, unpermute, index_select, reduce]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, vector_core]
architectures: [Ascend910B1, Ascend910B2, Ascend910_93, Ascend950PR, Ascend950DT]
kernel_types: [moe]
operators: [moe_token_permute_grad]
symptoms: [precision_mismatch, ub_overflow]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/moe-token-permute-grad
related: [kernel-moe-finalize-routing-v2-grad]
sources: [ops-transformer-moe-token-permute-grad]
performance_claims: []
---

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

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、950 代理正确性需结合源码/硬件进一步验证。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 [[ops-transformer-moe-token-permute-grad]]。