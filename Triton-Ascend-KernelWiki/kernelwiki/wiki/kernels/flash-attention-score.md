---
id: kernel-flash-attention-score
title: FlashAttentionScore training-forward on Ascend NPU
type: kernel
tags: [flash_attention, attention, softmax, online_softmax, matmul, cube_vector_pipe]
techniques: [fp32_accumulation, tail_mask]
hardware_features: [unified_buffer, l0c_to_ub, cube_vector_1_to_2, l1_buffer, l0c_buffer]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR]
kernel_types: [attention]
operators: [flash_attention_score, flash_attention_score_grad, prompt_flash_attention, incre_flash_attention, sparse_flash_attention]
symptoms: [precision_mismatch, ub_overflow, performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/flash-attention-score
related: [technique-flash-attention-tiling, technique-flash-attention-pipeline, pattern-flash-attention-tiling-key-dispatch]
sources: [ops-transformer-flash-attention-score]
performance_claims: []
indicators: [indicator-01, indicator-03, indicator-04, indicator-10, indicator-11, indicator-14, indicator-15, indicator-16, indicator-18]
---

# FlashAttentionScore (训练前向)

CANN ops-transformer 中的 `FlashAttentionScore` 算子在训练场景下使用 FlashAttention 算法实现 self-attention 前向计算，支持 Atlas A2 训练系列（arch32）与 Ascend 950PR/950DT（arch35）。本页归纳算子语义与计算流程，可复用的 tiling/流水/dispatch 知识分别见 [[technique-flash-attention-tiling]]、[[technique-flash-attention-pipeline]]、[[pattern-flash-attention-tiling-key-dispatch]]。

## 1. 输入输出

### 输入（按 op_host/flash_attention_score_def.cpp 顺序）

| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| query | 必选 | 查询，shape 由 input_layout 决定 | FP16/BF16/FLOAT/FP8_E5M2/FP8_E4M3FN/HIFLOAT8 |
| key | 必选 | 键 | 同 query |
| value | 必选 | 值 | 同 query |
| real_shift (pse) | 可选 | 位置编码 | FP16/BF16/FLOAT |
| drop_mask | 可选 | dropout 掩码，1=保留 0=丢弃 | UINT8 |
| padding_mask | 可选 | padding 掩码 | FP16/BF16/FLOAT |
| atten_mask | 可选 | 注意力掩码，1=不参与 0=参与 | BOOL/UINT8 |
| prefix | 可选 | prefix 序列长度 | INT64 |
| actual_seq_qlen / actual_seq_kvlen | 可选 | 变长真实序列长度 | INT64 |
| q_start_idx / kv_start_idx | 可选 | 序列起始偏移 | INT64 |
| d_scale_q / d_scale_k / d_scale_v | 可选 | FP8 全量化参数 | FLOAT |
| query_rope / key_rope | 可选 | RoPE 编码张量 | 同 query |
| sink | 可选 | attention sink | FLOAT |
| p_scale | 可选 | P 矩阵缩放 | FLOAT |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| softmax_max | Softmax 行最大值，供反向 | FLOAT |
| softmax_sum | Softmax 行求和，供反向 | FLOAT |
| softmax_out | Softmax 后的 P 矩阵 | FP16/BF16/FLOAT |
| attention_out | 最终注意力输出 | FP16/BF16/FLOAT |

### 维度约定

以 BNSD/BSND layout 为例（BSH/SBH 下 H=N*D）:
- B: 1~2M（prefix 时最大 2K）
- N: 1~256
- S: 1~1M
- D: 1~768（FP8 时 1~128）

内核计算逻辑轴为 B、N2（kv head）、G（query/kv head 比）、S1（query seq）、S2（kv seq）、D（head dim）、Dv（value head dim）。

### 属性

- `scale_value` (默认 1.0): QK 缩放系数
- `keep_prob` (默认 1.0): dropout 保留概率
- `head_num` (必需): query head 数
- `input_layout` (必需): BSH/SBH/BNSD/BSND/TND
- `sparse_mode` (默认 0): 0=ALL,1=NONE,2=ANY,3=CAUSAL,4=BAND,5=PREFIX,...
- `pse_type` (默认 1): 控制 pse 的 add/mul 顺序
- `inner_precise` (默认 0): 内部精度模式
- `pre_tockens` / `next_tockens`: sparse 窗口

## 2. 核心计算公式

pseType=1 时（先 add 再 mul）:
```
attention_out = Dropout(Softmax(Mask(scale * (pse + (Q·dScaleQ)·(K·dScaleK)ᵀ), atten_mask)), keep_prob) · (V·dScaleV)
```

pseType≠1 时（先 mul 再 add）:
```
attention_out = Dropout(Softmax(Mask(scale * ((Q·dScaleQ)·(K·dScaleK)ᵀ) + pse), atten_mask)), keep_prob) · (V·dScaleV)
```

FP8 场景下 dScaleQ/K/V 为全量化参数，非 FP8 时可视为 1。atten_mask 中为 true 的位置被 select 成负极小值，softmax 后变 0 实现遮蔽。

## 3. 计算流程（FlashSoftmax 在线刷新）

| 阶段 | 入口函数 | 单元 | 操作 |
|------|----------|------|------|
| Cube1 | IterateBmm1 | AIC | Q·Kᵀ matmul → 原始 attention_score |
| Vector1 | ProcessVec1 | AIV | pse add/mul → scale → atten_mask select → FlashSoftmax(在线 max/sum) |
| Cube2 | IterateBmm2 | AIC | P·V matmul（P 为 softmax 结果） |
| Vector2 | ProcessVec2 | AIV | rescale（除 softmax_sum）→ dropout → 写回 attention_out |

FlashSoftmax 对 Skv 轴切分，在线更新:
- `max[i]` = 第 i 个 SkvSplit 的行最大值
- `exp[i] = e^(max[i-1] - max[i])`
- `PV[i] = PV[i-1] * exp[i] + PV_new[i]`（PV 累积器按 exp 缩放后累加新分块）
- 最终 `attention_out = PV_final / softmax_sum`

softmax_max 与 softmax_sum 作为中间结果输出，供反向 `FlashAttentionScoreGrad` 使用。

## 4. 变体对照

| 变体 | 算子名 | 场景 |
|------|--------|------|
| 训练前向 | flash_attention_score | self-attention 训练，输出 softmax_max/sum 供反向 |
| 训练反向 | flash_attention_score_grad | 训练反向梯度 |
| 推理前向 | prompt_flash_attention | 推理 prefill / 全序列 |
| 增量推理 | incre_flash_attention | 推理增量 decode（KV cache） |
| 稀疏 | sparse_flash_attention / kv_quant_sparse_flash_attention | 稀疏 / KV 量化注意力<br>- MLA-absorb 模式（attention_mode=2）：query/key 的 nope 和 rope 部分沿 D 维拼接<br>- 离散访存搬运聚合优化（sparse_indices 驱动）<br>- 约束：D=512, Dr=64, N1=1/2/4/8/16/32/64/128, KV_N 只支持 1, block_size=16 倍数(≤1024)<br>- 支持 BSND/TND/PA_BSND 三种 layout<br>- 支持 FP16/BF16 输入输出，FP32 中间计算 |

所有变体共享 QK→softmax→PV 的核心计算骨架，差异在于 mask/sparse 处理、KV cache 管理、是否输出反向中间量。

## 5. 架构与产品支持

| 产品 | 架构代号 | 支持 | 说明 |
|------|----------|------|------|
| Atlas A2 训练 | arch32 (DAV_A2) | √ | AIC&AIV 分离，多模板（6 类） |
| Atlas A2 推理 | arch32 | √ | SFA 场景，稀疏 / KV 量化注意力 |
| Atlas A3 训练 | ascend910_93 | √ | 复用 arch32 kernel |
| Ascend 950PR/950DT | arch35 (DAV_3510) | √ | AIC&AIV 融合，单模板覆盖全 shape |
| Atlas A3 推理 | arch35 | √ | SFA 场景，稀疏 / KV 量化注意力 |

FP8（FLOAT8_E5M2/E4M3FN/HIFLOAT8）仅在 arch35 支持，且不可配 pse/dropMask/attenMask/keepProb 等可选参数。

SFA（sparse_flash_attention）仅支持推理场景，不支持训练。

## 6. Agent actions

1. 确认目标架构（arch32 vs arch35）与 dtype，选择对应模板分发路径。
2. 按输入 layout（BNSD/BSND/BSH/SBH/TND）和数据轴推导核内/核间切分轴。
3. 评估 S2 是否 >1024：是则 S1S2 模板（含 FlashSoftmax 刷新流程），否则 S1/SameAB 模板。
4. 检查 atten_mask/dropMask/pse 是否存在，选择对应 tiling key bit。
5. 参考 [[technique-flash-attention-pipeline]] 安排 Cube/Vector 流水与 double buffer。
6. Triton 实现参考 `artifacts/flash-attention-score/flash-attention-triton-skeleton.py`（pseudocode）。
7. 对于 sparse_flash_attention 算子，关注 attention_mode=2 的 MLA-absorb 模式，以及 sparse_indices 驱动的离散 KV 取数策略。参考 [[technique-sparse-kv-gather]] 了解搬运聚合优化，参考 [[technique-sparse-attention-tiling]] 了解 SFA 特有的 tiling 策略。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred），tiling 参数数值、流水 bound 结论未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 [[ops-transformer-flash-attention-score]]。
