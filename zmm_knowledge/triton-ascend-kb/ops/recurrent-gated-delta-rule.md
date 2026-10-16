---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU
description: CANN ops-transformer 的 RecurrentGatedDeltaRule（RGDR）是线性注意力机制算子，基于循环门控 Delta 规则（Recurrent Gated Delta Rule），以 SSM（状态空间模型）风格维护隐藏状态 St 并逐步更新。与 flash attention 系列不同，本算子为 AIV-only（无 Cube 参与），串行遍历 token（README L16-21）。
tags: [linear_attention, ssm, recurrent, gated_delta_rule, rnn, aiv_only, bf16, elementwise, matmul, fp32_accumulation, unified_buffer, ascend910b1, ascend910_93, attention, recurrent_gated_delta_rule]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/recurrent_gated_delta_rule/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/recurrent_gated_delta_rule/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/recurrent-gated-delta-rule.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/recurrent_gated_delta_rule/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/recurrent_gated_delta_rule/)。

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

本页结论基于源码静态分析（confidence: inferred）。dk/dv 精确约束、MTP 场景适用性、tiling 参数最优值未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 `ops-transformer-recurrent-gated-delta-rule`。

<!-- okf:related:start -->

# 相关

- 实践案例: [MatMul / GEMM 高性能算子](../reference/op-class-matmul.md) — 主题相关：共享 matmul
- 实践案例: [SSM / 状态空间模型算子优化](../reference/op-class-ssm.md) — 主题相关：共享 linear_attention, recurrent, ssm
- 实践案例: [matmul core mapping 策略选择（列/行/对角）](../runbooks/compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 实践案例: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../runbooks/compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 实践案例: [计算优化: fixpipe 随路量化 + K 维循环展开 + 预取](../runbooks/compute-pipeline/exp_matmul_002.md) — 主题相关：共享 matmul
- 实践案例: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 实践案例: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../runbooks/memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 实践案例: [Add算子块大小动态调整优化经验](../runbooks/memory-sram/exp_elem_add_001_20260526.md) — 主题相关：共享 elementwise
- 实践案例: [带 alpha 缩放的逐元素 add 访存与块大小优化](../runbooks/memory-sram/exp_elemwise_add_001_20260526.md) — 主题相关：共享 elementwise
- 实践案例: [访存优化：大尺寸 matmul 的 256 分块与 L0C 复用](../runbooks/memory-sram/exp_matmul_001.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块](../runbooks/memory-sram/exp_matmul_003.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块](../runbooks/memory-sram/exp_matmul_004.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：双转置矩阵乘的 512B 对齐分块](../runbooks/memory-sram/exp_matmul_005.md) — 主题相关：共享 matmul
- 实践案例: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](../runbooks/memory-sram/layout-bthd-to-bhtd.md) — 主题相关：共享 linear_attention
- 实践案例: [Cube-Vector 场景行宽必须 512B 对齐](../runbooks/memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 实践案例: [tl.dot vs torch.matmul 精度差异诊断](../runbooks/meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡](add-broadcast-alpha.md) — 主题相关：共享 aiv_only, bf16, elementwise
- 相关主题: [ApplyRotaryPosEmb（RoPE 旋转位置编码 q/k 双路融合）Ascend C → Triton 语义转化卡](apply-rotary-pos-emb.md) — 主题相关：共享 aiv_only, bf16, elementwise
- 相关主题: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](chunk-gated-delta-rule-fwd-h.md) — 同 gated delta rule 算子族：RGDR 逐步递推 vs FwdH chunkwise 并行形式
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 aiv_only, ascend910b1, bf16, elementwise, fp32_accumulation
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, bf16, elementwise, fp32_accumulation
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](fused-bias-leaky-relu.md) — 主题相关：共享 aiv_only, elementwise
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [GDN 线性注意力三阶段优化案例（FLA 移植 → Triton 体系化 → AscendC 重写）](gdn-chunked-linear-attention.md) — 主题相关：共享 linear_attention
- 相关主题: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](gdn2-fwd-chunked-wy-dispatch.md) — 同 GDN 算子族：RGDR 为逐步形式，Gdn2Fwd 为 chunkwise WY 形式
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, bf16, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, attention, bf16, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 aiv_only, ascend910_93, ascend910b1, attention, bf16, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, bf16, fp32_accumulation, unified_buffer

<!-- okf:related:end -->
