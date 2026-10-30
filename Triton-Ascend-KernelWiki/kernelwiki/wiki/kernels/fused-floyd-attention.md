---
id: kernel-fused-floyd-attention
title: FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU
type: kernel
tags: [floyd_attention, attention, multi_dim, dual_kv, softmax, matmul, cube_vector_pipe]
techniques: [fp32_accumulation, tail_mask]
hardware_features: [unified_buffer, l0c_buffer, l1_buffer, l0c_to_ub, cube_vector_1_to_2]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [attention]
operators: [fused_floyd_attention]
symptoms: [performance_bottleneck, precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/fused-floyd-attention
related: [technique-flash-attention-tiling, technique-flash-attention-pipeline]
sources: [ops-transformer-fused-floyd-attention]
performance_claims: []
---

# FusedFloydAttention (Floyd 多维注意力)

CANN ops-transformer 中的 `FusedFloydAttention` 算子使用 FloydAttention 算法实现多维自注意力计算，支持 2 组 KV head group（K1/V1 + K2/V2）。仅支持 Atlas A2 训练系列（arch32 / ascend910b / ascend910_93），不支持 ascend950。

## 1. 输入输出

### 输入
| 输入 | 必选 | 含义 | dtype |
|------|------|------|-------|
| query | 是 | 查询 Q | FP16/BF16/FLOAT |
| key1 | 是 | 键 K1 | FP16/BF16/FLOAT |
| value1 | 是 | 值 V1 | FP16/BF16/FLOAT |
| key2 | 是 | 键 K2 | FP16/BF16/FLOAT |
| value2 | 是 | 值 V2 | FP16/BF16/FLOAT |
| atten_mask | 可选 | 注意力掩码 | BOOL/UINT8 |

### 输出
| 输出 | 含义 | dtype |
|------|------|-------|
| softmax_max | Softmax 行最大值 | FLOAT |
| softmax_sum | Softmax 行求和 | FLOAT |
| attention_out | 注意力输出 | FP16/BF16/FLOAT |

### 约束（README.md L112-124）
B:1~2K, H:1~256, N:16~1M(N%16==0), M:128~1M(M%128==0), K:128~1M(K%128==0), D:32/64/128

## 2. 核心公式

```
weights = Softmax(attenMask + scale * (Q@K1^T + Q@K2^T))
attention_out = weights@V1 + weights@V2
```

4 次 matmul: Q·K1^T + Q·K2^T + P·V1 + P·V2

## 3. 计算流程

| 阶段 | 入口 | 单元 | 操作 |
|------|------|------|------|
| Cube1 | IterateBmm1 | AIC | Q·K1^T |
| Cube1k1 | IterateBmm1K1 | AIC | Q·K2^T |
| Vector1 | ProcessVec1 | AIV | QK1+QK2 → scale → mask → Softmax |
| Cube2 | IterateBmm2 | AIC | P·V1 |
| Cube2v2 | IterateBmm2V2 | AIC | P·V2 |
| Vector2 | ProcessVec2 | AIV | P·V1+P·V2 → 写回 |

## 4. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 [[ops-transformer-fused-floyd-attention]]。