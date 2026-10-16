---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: FlashAttentionScore training-forward on Ascend NPU
description: CANN ops-transformer 中的 FlashAttentionScore 算子在训练场景下使用 FlashAttention 算法实现 self-attention 前向计算，支持 Atlas A2 训练系列（arch32）与 Ascend 950PR/950DT（arch35）。本页归纳算子语义与计算流程，可复用的 tiling/流水/dispatch 知识分别见 technique-flash-attention-tiling、technique-fl...
tags: [flash_attention, attention, softmax, online_softmax, matmul, cube_vector_pipe, fp32_accumulation, tail_mask, unified_buffer, l0c_to_ub, cube_vector_1_to_2, l1_buffer, l0c_buffer, ascend910b1, ascend910_93, ascend950pr, flash_attention_score, flash_attention_score_grad, prompt_flash_attention, incre_flash_attention, sparse_flash_attention]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/flash_attention_score/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/flash_attention_score/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/flash-attention-score.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/flash_attention_score/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/flash_attention_score/)。

# FlashAttentionScore (训练前向)

CANN ops-transformer 中的 `FlashAttentionScore` 算子在训练场景下使用 FlashAttention 算法实现 self-attention 前向计算，支持 Atlas A2 训练系列（arch32）与 Ascend 950PR/950DT（arch35）。本页归纳算子语义与计算流程，可复用的 tiling/流水/dispatch 知识分别见 `technique-flash-attention-tiling`、`technique-flash-attention-pipeline`、`pattern-flash-attention-tiling-key-dispatch`。

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
5. 参考 `technique-flash-attention-pipeline` 安排 Cube/Vector 流水与 double buffer。
6. Triton 实现参考 `artifacts/flash-attention-score/flash-attention-triton-skeleton.py`（pseudocode）。
7. 对于 sparse_flash_attention 算子，关注 attention_mode=2 的 MLA-absorb 模式，以及 sparse_indices 驱动的离散 KV 取数策略。参考 `technique-sparse-kv-gather` 了解搬运聚合优化，参考 `technique-sparse-attention-tiling` 了解 SFA 特有的 tiling 策略。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred），tiling 参数数值、流水 bound 结论未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 `ops-transformer-flash-attention-score`。

**已知实例（硬件实测）**：CrossformerAttention（attention_standard_flash，ascend910b，2026-10-10）——以本卡为 CANN 参考（flash_attention_score d28d8f99a），逐阶段对照 Triton flash kernel 计算流程差异：① scale 融合位置不同（CANN 早乘进 Q 于 cube QKᵀ 阶段；Triton 于 dot 结果后 vector 乘，数学等价、fp32 位模式略异且容差内）；② CANN 依赖 arch35 L1 3-buffer + D-workspace + regbase 深流水把 BLOCK 拉到 128/128，而 Triton 单 kernel 在同一 192KB UB 下只能退化为小 BM/BN（大 D(>256) 无 D-split），此为 speedup_vs_aclnn 仅 0.0043（~1160x 差距）的结构性来源。差异明细见工作目录 `triton_ascend_output/op_0_15_CrossformerAttention_20261010_1653_8516/ir_diff_log.md` D1-D9。

<!-- okf:related:start -->

# 相关

- 实践案例: [Fused Attention / Flash Attention 高性能算子](../reference/op-class-attention.md) — 主题相关：共享 attention, flash_attention, softmax
- 实践案例: [MatMul / GEMM 高性能算子](../reference/op-class-matmul.md) — 主题相关：共享 matmul
- 实践案例: [归一化算子优化](../reference/op-class-normalization.md) — 主题相关：共享 softmax
- 实践案例: [Softmax / LogSoftmax 算子优化](../reference/op-class-softmax.md) — 主题相关：共享 online_softmax, softmax
- 实践案例: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 softmax
- 实践案例: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](../runbooks/algorithm-numerics/exp_stability_001.md) — 同属优化知识，主题相关
- 实践案例: [Online Softmax / Flash Attention 单遍稳定计算](../runbooks/algorithm-numerics/online-softmax-flash-attention.md) — 主题相关：共享 flash_attention, online_softmax
- 实践案例: [matmul core mapping 策略选择（列/行/对角）](../runbooks/compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 实践案例: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../runbooks/compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 实践案例: [流程优化: 标准 score-归一化-加权三步流程](../runbooks/compute-pipeline/exp_fused_attention_001.md) — 主题相关：共享 attention
- 实践案例: [Scheduling Optimization: hardware unit isolation](../runbooks/compute-pipeline/exp_fused_norm_rope_matmul_001.md) — 主题相关：共享 matmul
- 实践案例: [计算优化: fixpipe 随路量化 + K 维循环展开 + 预取](../runbooks/compute-pipeline/exp_matmul_002.md) — 主题相关：共享 matmul
- 实践案例: [Compute Optimization: vectorized accumulation strategy](../runbooks/compute-pipeline/exp_softmax_inner_002.md) — 主题相关：共享 softmax
- 实践案例: [Compute Optimization: online softmax algorithm](../runbooks/compute-pipeline/exp_softmax_non_inner_001.md) — 主题相关：共享 softmax
- 实践案例: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../runbooks/memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 实践案例: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](../runbooks/memory-sram/exp_fused_attention_002.md) — 主题相关：共享 attention
- 实践案例: [访存优化：大尺寸 matmul 的 256 分块与 L0C 复用](../runbooks/memory-sram/exp_matmul_001.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块](../runbooks/memory-sram/exp_matmul_003.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块](../runbooks/memory-sram/exp_matmul_004.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：双转置矩阵乘的 512B 对齐分块](../runbooks/memory-sram/exp_matmul_005.md) — 主题相关：共享 matmul
- 实践案例: [Memory Access Optimization: score-tile reuse with online softmax](../runbooks/memory-sram/exp_sdpa_flash_001.md) — 主题相关：共享 attention
- 实践案例: [Memory Access Optimization: two pass reduction strategy](../runbooks/memory-sram/exp_softmax_backward_001.md) — 主题相关：共享 softmax
- 实践案例: [Memory Access Optimization: two pass reduction strategy](../runbooks/memory-sram/exp_softmax_backward_non_inner_001.md) — 主题相关：共享 softmax
- 实践案例: [Cube-Vector 场景行宽必须 512B 对齐](../runbooks/memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 实践案例: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../runbooks/meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [tl.dot vs torch.matmul 精度差异诊断](../runbooks/meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 实践案例: [禁止在 forward() 中使用 PyTorch 计算（Type-3 退化）](../runbooks/portability/no-pytorch-in-forward.md) — 同属优化知识，主题相关
- 相关主题: [Flash Attention Causal Skeleton](../reference/skeleton-flash_attention_causal.md) — 同属优化知识，主题相关
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, flash_attention, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 ascend910b1, ascend950pr, fp32_accumulation, tail_mask
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 同一 FlashAttentionScore 算子的前向与反向实现
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, matmul, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, softmax, tail_mask, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, flash_attention, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, cube_vector_pipe, flash_attention, fp32_accumulation, incre_flash_attention, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, flash_attention, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, matmul, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, softmax, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, softmax, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, tail_mask, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — FlashAttentionScore 与 prompt 场景变体同属 attention 家族
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, sparse_flash_attention, tail_mask, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, online_softmax, softmax, tail_mask, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, fp32_accumulation, online_softmax, softmax, unified_buffer
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, flash_attention, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, online_softmax, softmax, sparse_flash_attention, tail_mask, unified_buffer

<!-- okf:related:end -->
