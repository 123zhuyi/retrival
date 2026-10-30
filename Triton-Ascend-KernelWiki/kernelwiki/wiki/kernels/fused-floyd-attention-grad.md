---
id: kernel-fused-floyd-attention-grad
title: FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU
type: kernel
tags: [fused_floyd_attention_grad, floyd_attention, attention, backward, matmul, training, kimi]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, l0c_buffer, cube_vector_1_to_2]
architectures: [Ascend910B1, Ascend910B2, Ascend910_93]
kernel_types: [attention]
operators: [fused_floyd_attention_grad]
symptoms: [precision_mismatch, ub_overflow]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/fused-floyd-attention-grad
related: [technique-flash-attention-tiling, technique-flash-attention-pipeline, kernel-flash-attention-score]
sources: [ops-transformer-fused-floyd-attention-grad]
performance_claims: []
---

# FusedFloydAttentionGrad (Floyd attention backward)

CANN ops-transformer 中的 `FusedFloydAttentionGrad` 算子是 **Floyd Attention（Moonshot Kimi 长上下文双流注意力）** 的反向梯度算子。双流结构（K1/V1 + K2/V2 共享 Q），训练场景，仅 Atlas A2/A3 训练系列（arch32）。核心为 6 个 matmul 反向计算 dQ/dK1/dV1/dK2/dV2，三段式 kernel（Pre → S1S2 → Post）。

## 1. 输入输出

### 输入（按 `fused_floyd_attention_grad_def.cpp`）

| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| query | 必选 | 前向 Q | FP16/BF16/FLOAT |
| key1 | 必选 | 前向 K1 | FP16/BF16/FLOAT |
| value1 | 必选 | 前向 V1 | FP16/BF16/FLOAT |
| key2 | 必选 | 前向 K2 | FP16/BF16/FLOAT |
| value2 | 必选 | 前向 V2 | FP16/BF16/FLOAT |
| dy | 必选 | 下游梯度 dY | FP16/BF16/FLOAT |
| atten_mask | 可选 | 注意力掩码 | BOOL/UINT8 |
| softmax_max | 可选 | 前向 softmax 行 max | FLOAT |
| softmax_sum | 可选 | 前向 softmax 行 sum | FLOAT |
| attention_in | 可选 | 前向 attention_out | FP16/BF16/FLOAT |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| dq | query 梯度 | FP16/BF16/FLOAT |
| dk_1 | key1 梯度 | FP16/BF16/FLOAT |
| dv_1 | value1 梯度 | FP16/BF16/FLOAT |
| dk_2 | key2 梯度 | FP16/BF16/FLOAT |
| dv_2 | value2 梯度 | FP16/BF16/FLOAT |

### 属性

`scale_value`（默认 1.0）：缩放系数。

### 维度约束（README L173-189）

- B：1~2K，H：1~256，N：16~1M（N%16==0），M：128~1M（M%128==0），K：128~1M（K%128==0）
- D 仅支持 32/64/128
- key1 与 value1 shape 相同，key2 与 value2 shape 相同

## 2. 核心计算公式

```
前向：P = Softmax(scale * (Q·K1^T + Q·K2^T), mask)
反向：
  dS  = S * dY              # 下游梯度
  dV1 = P^T · dY            # Value1 梯度
  dV2 = P^T · dY            # Value2 梯度
  dQ  = (dS·K1 + dS·K2) / √d  # Query 梯度（双流相加）
  dK1 = (dS^T · Q) / √d     # Key1 梯度
  dK2 = (dS^T · Q) / √d     # Key2 梯度
```

## 3. 计算流程

### 3.1 三段式 kernel

```
Pre 阶段：dS = S * dY（利用前向保存的 softmax_max/sum 恢复 S）
  → SyncALLCores()
S1S2 阶段：
  → Bmm1: dV1 = P^T · dY
  → Bmm2: dV2 = P^T · dY
  → MM1:  dQ = (dS·K1 + dS·K2) / √d
  → MM2:  dK1 = (dS^T · Q) / √d
  → MM3:  dK2 = (dS^T · Q) / √d
  → SyncALLCores()
Post 阶段：梯度搬运 + ND↔NZ 格式转换
```

### 3.2 双流注意力特色

Floyd Attention 将 seq 作为 batch 轴，将 Q·K^T 分解为 Q·K1^T + Q·K2^T 两部分：
- K1/V1 和 K2/V2 共享 Q
- dQ = dQ_K1 + dQ_K2（双流相加）
- dK1 和 dK2 各自独立计算（公式对称）

## 4. 模板与 dispatch

| dtype | TilingKey | mask | pse | drop |
|-------|-----------|------|-----|------|
| BF16 | 10000000000000022434UL | 0 | 0 | 0 |
| BF16 | 10000000000100022434UL | 1 | 0 | 0 |
| FP16 | 10000000000000023434UL | 0 | 0 | 0 |
| FP16 | 10000000000100023434UL | 1 | 0 | 0 |

## 5. 架构与产品支持

| 产品 | 架构 | config（def.cpp） | kernel 入口 |
|------|------|-------------------|-------------|
| Atlas A2 训练 | arch32 | `ascend910b` L164 | `fused_floyd_attention_grad.cpp` |
| Atlas A3 训练 | ascend910_93 | `ascend910_93` L165 | 同上 |

> 仅训练场景，无 950，无 310P。支持 FP16/BF16/FLOAT（FLOAT 仅 A2/A3 训练）。

## 6. Agent actions

1. 确认目标架构为 A2/A3 训练（arch32），dtype（FP16/BF16/FLOAT）。
2. 6 个 matmul 反向映射：`tl.dot` × 6（dV1/dV2/dQ/dK1/dK2）。
3. 双流处理：dQ = `tl.dot(dS,K1) + tl.dot(dS,K2)` 后 `/= sqrt(d)`。
4. 参考 [[technique-flash-attention-tiling]] / [[technique-flash-attention-pipeline]] 了解 FA 通用 tiling/流水知识。
5. Triton 实现参考 `artifacts/fused-floyd-attention-grad/fused-floyd-attention-grad-triton-skeleton.py`（pseudocode）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、dS 精确公式（依赖前向 softmax_max/sum 恢复）、ND↔NZ 格式转换细节需结合源码/硬件进一步验证。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 [[ops-transformer-fused-floyd-attention-grad]]。