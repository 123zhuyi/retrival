---
id: kernel-ring-attention-update
title: RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU
type: kernel
tags: [ring_attention, attention, softmax, online_softmax, elementwise, aiv_only, fp16, bf16, fp32]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR, Kirin9030, KirinX90]
kernel_types: [attention]
operators: [ring_attention_update]
symptoms: [precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/ring-attention-update
related: [kernel-flash-attention-score, kernel-prompt-flash-attention, technique-flash-attention-pipeline]
sources: [ops-transformer-ring-attention-update]
performance_claims: []
---

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