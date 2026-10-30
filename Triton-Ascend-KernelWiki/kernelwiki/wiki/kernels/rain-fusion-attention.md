---
id: kernel-rain-fusion-attention
title: RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU
type: kernel
tags: [rain_fusion, sparse_attention, attention, softmax, online_softmax, matmul, catlass, block_sparse, cube_vector_pipe]
techniques: [fp32_accumulation, tail_mask, sparse_attention_tiling]
hardware_features: [unified_buffer, l0c_to_ub, cube_vector_1_to_2, l1_buffer, l0c_buffer]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [attention]
operators: [rain_fusion_attention]
symptoms: [performance_bottleneck, precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/rain-fusion-attention
related: [kernel-sparse-flash-attention, kernel-flash-attention-score, technique-sparse-attention-tiling, technique-sparse-attention-pipeline]
sources: [ops-transformer-rain-fusion-attention]
performance_claims: []
---

# RainFusionAttention（CATLASS 块级稀疏推理注意力）

CANN ops-transformer 的 `RainFusionAttention` 是基于 **CATLASS 模板库**的高性能块级稀疏推理注意力算子（从 catlass `32_sparse_attention_infer` 迁移）。与 [[kernel-sparse-flash-attention]] 一样使用稀疏 QK/PV，但稀疏模式由 `selectIdx`（显式 Q×KV 块选择矩阵）驱动，比 SFA 的 `sparse_indices` 更灵活（README L138-163）。**注意**：README 说 BNSD 不支持，但源码 tilingkey.h 和 kernel 入口定义了 BNSD 分支——以源码为准（`needs_verification`）。

## 1. 输入输出

### 输入（def.cpp L19-58）

| 输入 | 必选/可选 | 含义 | dtype | format |
|---|---|---|---|---|
| query | 必选 | Q，[B,S,H]/[T,N,D] | FP16/BF16 | ND |
| key / value | 必选 | K/V，仅 TND | FP16/BF16 | ND |
| selectIdx | 必选 | 稀疏块索引 [T,headNum,maxKvBlockNum] | INT64 | ND |
| selectNumIdx | 必选 | 每 Q 块的 KV 块数 [T,headNum] | INT64 | ND |
| blockShape | 必选 | 块形状 [blockShapeX, blockShapeY] | INT64 | ND |
| attenMask | 可选 | 注意力掩码 | FP16/BF16 | ND |
| actualSeqLengths / _kv | 可选 | 实际序列长度 | INT64 | ND |
| blockTable | 可选 | PagedAttention 块表 | INT32 | ND |

### 输出（def.cpp L59-66）
- attentionOut（REQUIRED，FP16/BF16）
- softmaxLse（OPTIONAL，FLOAT）

### 属性（def.cpp L68-74）
`qInputLayout`/`kvInputLayout`("TND")、`numKeyValueHeads`(1)、`maskType`(0)、`scaleValue`(0.0)、`innerPrecise`(1)、`blockSize`(128)。

## 2. 核心公式

```
for each Q block (blockShapeX):
  for each selected KV block (selectIdx):
    s = Q_block · K_block^T × scale
    s = mask(s)     # 可选
    # online softmax + PV rescale (同 FlashSoftmax)
    m_ij = max(s);  p = exp(s - m_ij);  l_ij = sum(p)
    alpha = exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha + p · V_block
    m_i = max(m_i, m_ij)
  o = acc / l_i;  lse = log(l_i) + m_i   # 可选
```

## 3. tiling key dispatch（tilingkey.h L31-46）

6 个 key：TND/BNSD × FP16/BF16 × FloatSM/HalfSM。base=`9000000000000000`。MIX_AIC_1_2。

## 4. 架构支持（def.cpp L76-77）

仅 Ascend910B1/910_93（A2/A3）。编译 `--soc=Ascend910B3`。不支持 950/Kirin。

## 5. Agent actions

1. 识别为 CATLASS 风格块级稀疏推理注意力，selectIdx 驱动 KV 块选择。
2. blockShapeX/Y 为稀疏块大小（constexpr BLOCK_M/N）。
3. 稀疏遍历与 SFA 类似但索引更灵活（显式 Q×KV 选择矩阵）。
4. innerPrecise 控制 softmax 精度（0=FP32/1=Half）。
5. softmaxLse 可选输出供下游。
6. BNSD layout 源码支持但 README 说不支持——以源码为准。

## 6. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。BNSD 实际支持状态（源码 vs README 矛盾）需验证。Triton skeleton 标记 `reproducibility: pseudocode`。