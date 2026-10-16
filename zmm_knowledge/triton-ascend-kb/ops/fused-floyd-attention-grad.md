---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU
description: CANN ops-transformer 中的 FusedFloydAttentionGrad 算子是 Floyd Attention（Moonshot Kimi 长上下文双流注意力） 的反向梯度算子。双流结构（K1/V1 + K2/V2 共享 Q），训练场景，仅 Atlas A2/A3 训练系列（arch32）。核心为 6 个 matmul 反向计算 dQ/dK1/dV1/dK2/dV2，三段式 kernel（Pre → S1S2 → Post）。
tags: [fused_floyd_attention_grad, floyd_attention, attention, backward, matmul, training, kimi, fp32_accumulation, unified_buffer, l0c_buffer, cube_vector_1_to_2, ascend910b1, ascend910b2, ascend910_93]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/fused_floyd_attention_grad/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/fused_floyd_attention_grad/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/fused-floyd-attention-grad.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/fused_floyd_attention_grad/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/fused_floyd_attention_grad/)。

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
4. 参考 `technique-flash-attention-tiling` / `technique-flash-attention-pipeline` 了解 FA 通用 tiling/流水知识。
5. Triton 实现参考 `artifacts/fused-floyd-attention-grad/fused-floyd-attention-grad-triton-skeleton.py`（pseudocode）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、dS 精确公式（依赖前向 softmax_max/sum 恢复）、ND↔NZ 格式转换细节需结合源码/硬件进一步验证。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 `ops-transformer-fused-floyd-attention-grad`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Embedding / 索引拷贝算子优化](../reference/op-class-embedding.md) — 主题相关：共享 backward
- 实践案例: [MatMul / GEMM 高性能算子](../reference/op-class-matmul.md) — 主题相关：共享 matmul
- 实践案例: [RoPE（旋转位置编码）算子优化](../reference/op-class-rope.md) — 主题相关：共享 backward
- 实践案例: [matmul core mapping 策略选择（列/行/对角）](../runbooks/compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 实践案例: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../runbooks/compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 实践案例: [Scheduling Optimization: hardware unit isolation](../runbooks/compute-pipeline/exp_fused_norm_rope_matmul_001.md) — 主题相关：共享 matmul
- 实践案例: [计算优化: fixpipe 随路量化 + K 维循环展开 + 预取](../runbooks/compute-pipeline/exp_matmul_002.md) — 主题相关：共享 matmul
- 实践案例: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../runbooks/memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 实践案例: [访存优化：大尺寸 matmul 的 256 分块与 L0C 复用](../runbooks/memory-sram/exp_matmul_001.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块](../runbooks/memory-sram/exp_matmul_003.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块](../runbooks/memory-sram/exp_matmul_004.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：双转置矩阵乘的 512B 对齐分块](../runbooks/memory-sram/exp_matmul_005.md) — 主题相关：共享 matmul
- 实践案例: [Cube-Vector 场景行宽必须 512B 对齐](../runbooks/memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 实践案例: [tl.dot vs torch.matmul 精度差异诊断](../runbooks/meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, backward, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, backward, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, matmul, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, matmul, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 同一 FusedFloydAttention 算子的前向与反向实现
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, matmul, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, matmul, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, matmul, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, backward, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, attention, backward, fp32_accumulation, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, training, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, matmul, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — 主题相关：共享 ascend910b1
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer

<!-- okf:related:end -->
