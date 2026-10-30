---
id: kernel-recurrent-gated-delta-rule
title: RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU
type: kernel
tags: [linear_attention, ssm, recurrent, gated_delta_rule, rnn, aiv_only, bf16, elementwise, matmul]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [attention]
operators: [recurrent_gated_delta_rule]
symptoms: [precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/recurrent-gated-delta-rule
related: [kernel-flash-attention-score, kernel-sparse-flash-attention]
sources: [ops-transformer-recurrent-gated-delta-rule]
performance_claims: []
---

# RecurrentGatedDeltaRule（线性注意力 / SSM 风格循环门控 Delta 规则）

CANN ops-transformer 的 `RecurrentGatedDeltaRule`（RGDR）是**线性注意力机制**算子，基于循环门控 Delta 规则（Recurrent Gated Delta Rule），以 SSM（状态空间模型）风格维护隐藏状态 `S_t` 并逐步更新。**与 flash attention 系列不同，本算子为 AIV-only（无 Cube 参与），串行遍历 token**（README L16-21）。

## 1. 输入输出

### 输入（def.cpp L22-71）

| 输入 | 必选/可选 | 含义 | dtype | format |
|---|---|---|---|---|
| query | 必选 | q_t，[B,NK,DK] | BF16 | ND |
| key | 必选 | k_t，[B,NK,DK] | BF16 | ND |
| value | 必选 | v_t，[B,NV,DV] | BF16 | ND |
| beta | 必选 | β_t，门控系数 | BF16 | ND |
| state | 必选(输入/输出) | S_t，隐藏状态 | BF16 | ND |
| actual_seq_lengths | 必选 | 实际序列长度(前缀和) | INT32 | ND |
| ssm_state_indices | 必选 | SSM 状态索引 | INT32 | ND |
| g | 可选 | 衰减系数 α=e^g | FLOAT | ND |
| gk | 可选 | 衰减系数 αk=e^gk | FLOAT | ND |
| num_accepted_tokens | 可选 | 接受的 token 数 | INT32 | ND |

### 输出（def.cpp L73-81）
- out：o_t，[B,NV,DV]（BF16）
- state：S_t（REQUIRED，原地更新）

### 属性（def.cpp L82）
`scale_value`（默认1.0）。

### 约束
仅 BF16，仅 A2/A3。`dk`/`dv` 约束以 `docs/aclnnRecurrentGatedDeltaRule.md` 为准（`needs_verification`）。

## 2. 核心公式（README L24-31）

```
S_t = α·Diag(αk)·S_{t-1} + β·(v_t - α·Diag(αk)·S_{t-1}·k_t)·k^T_t
o_t = S_t · q_t / √d
```
其中 `α = exp(g)`，`αk = exp(gk)`。`S_t ∈ [DV, DK]`，`q_t,k_t ∈ [DK]`，`v_t,o_t ∈ [DV]`。

## 3. 计算流程

**AIV-only**（kernel L31：`KERNEL_TYPE_AIV_ONLY`），无 CV 流水。

| 阶段 | 操作 | 说明 |
|---|---|---|
| 1. Init | 读 tiling data，设全局 tensor 指针 | kernel.h L72 |
| 2. 门控 | exp(g)/exp(gk) | 元素级 |
| 3. 衰减 | α·Diag(αk)·S_{t-1} | 矩阵乘 |
| 4. 误差 | v_t - decayed·k_t | 向量点积+减 |
| 5. 更新 | S_t = decayed + β·(error)·k^T_t | 外积 |
| 6. 输出 | o_t = S_t·q_t·scale | 矩阵-向量乘 |

## 4. tiling 与 dispatch（详见 `tiling_dispatch_tables.md`）

AIV-only 分核。`sBlockNum`/`vStep` 控制 V 维分块。`BUFFER_NUM=1`（无 double buffer）。BF16 输入 FP32 中间。

## 5. 架构支持（README L7-12）

| 产品 | 架构 | 支持 |
|---|---|---|
| Atlas A2/A3 训练/推理 | Ascend910B1/910_93 | √ |
| 950/310P/训练系列 | - | × |

## 6. Agent actions

1. 识别为**线性注意力**（SSM 风格，非 QK matmul 主导），AIV-only。
2. 核心公式 `S_t = α·Diag(αk)·S_{t-1} + β·(v - α·Diag(αk)·S_{t-1}·k)·k^T`：Triton 中 `tl.dot` 处理矩阵乘 + `tl.where` 处理误差 + 外积更新。
3. 串行遍历 token（cu_seqlens 前缀和驱动），无 CUDA 式并行 S 轴。
4. 可选 g/gk/num_accepted_tokens 用 `if constexpr` 处理。
5. `BUFFER_NUM=1` 无 double buffer（CANN 特征）。
6. 参考 `artifacts/recurrent-gated-delta-rule/recurrent-gated-delta-rule-triton-skeleton.py`（pseudocode）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。dk/dv 精确约束、MTP 场景适用性、tiling 参数最优值未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 [[ops-transformer-recurrent-gated-delta-rule]]。