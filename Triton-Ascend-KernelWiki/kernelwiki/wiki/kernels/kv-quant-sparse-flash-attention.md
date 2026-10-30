---
id: kernel-kv-quant-sparse-flash-attention
title: KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU
type: kernel
tags: [kv_quant_sparse_flash_attention, sparse_attention, flash_attention, quant, kv_cache, mla, inference]
techniques: [fp32_accumulation, sparse_kv_gather, sparse_attention_tiling, sparse_attention_pipeline]
hardware_features: [unified_buffer, l0c_buffer, cube_vector_1_to_2]
architectures: [Ascend910B1, Ascend910B2, Ascend910_93, Ascend950PR, Ascend950DT]
kernel_types: [attention]
operators: [kv_quant_sparse_flash_attention]
symptoms: [precision_mismatch, ub_overflow, performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/kv-quant-sparse-flash-attention
related: [technique-sparse-attention-tiling, technique-sparse-attention-pipeline, technique-sparse-kv-gather, kernel-flash-attention-score]
sources: [ops-transformer-kv-quant-sparse-flash-attention]
performance_claims: []
---

# KvQuantSparseFlashAttention (sparse inference with KV quant)

CANN ops-transformer 中的 `KvQuantSparseFlashAttention` 算子是在 `sparse_flash_attention`（SFA）基础上支持 **Per-Token-Head-Tile-128 KV 量化**的推理稀疏注意力算子。仅推理场景（A2/A3 推理 + 950），核心为 sparse_indices 驱动的离散 KV 取数 + KV INT8/FP8 反量化 + MLA-absorb 模式（Q/K nope+rope 拼接 D=576/656）。稀疏策略/离散访存优化见 [[technique-sparse-kv-gather]]、[[technique-sparse-attention-tiling]]、[[technique-sparse-attention-pipeline]]。

## 1. 输入输出

### 输入（按 README.md L29-84）

| 输入 | 必选/可选 | 含义 | dtype | 说明 |
|------|-----------|------|-------|------|
| query | 必选 | Q（nope+rope 拼接 D=576） | BF16/FP16 | BSND/TND |
| key | 必选 | K（nope+rope+dequant_scale 拼接 D=656） | INT8（arch32）/ FP8_E4M3FN/HIFLOAT8（arch35） | BSND/TND/PA_BSND |
| value | 必选 | V（同 key 量化） | 同 key | 同 key |
| sparse_indices | 必选 | 离散取数索引 [B, Q_S, KV_N, sparse_size] | INT32 | - |
| scale_value | 必选 | 缩放系数 1/√d | float | - |
| key_quant_mode | 必选 | 仅支持 2（per-tile） | int64 | - |
| value_quant_mode | 必选 | 仅支持 2（per-tile） | int64 | - |
| block_table | 可选 | PA block 映射表 | INT32 | - |
| actual_seq_lengths_query/kv | 可选 | 变长序列长度 | INT32 | - |
| sparse_block_size | 可选 | sparse 块大小（1-16，2 幂次） | int64 | 默认 1 |
| layout_query/kv | 可选 | 数据排布格式 | str | 默认 BSND |
| sparse_mode | 可选 | 3=rightDownCausal | int64 | 默认 3 |
| attention_mode | 可选 | 2=MLA-absorb | int64 | 默认 0 |
| quant_scale_repo_mode | 可选 | 1=combine mode | int64 | 默认 1 |
| tile_size | 可选 | per-tile 量化块大小 | int64 | 默认 128 |
| rope_head_dim | 可选 | MLA rope head dim | int64 | 默认 64 |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| attention_out | 注意力输出 | BF16/FP16 |

### 约束

- D=576（nope 512 + rope 64）；K D=656（nope 512 + rope 128 + dequant_scale 16）
- KV_N 仅支持 1
- sparse_block_size 整除 block_size
- 非 PA 场景 layout_query 与 layout_kv 一致

## 2. 核心计算公式

```
Attention = softmax((Q · Dequant(K^INT8, Scale_K)^T) / √d) · Dequant(V^INT8, Scale_V)
```

## 3. 计算流程

### 3.1 离散 KV 取数（sparse_indices 驱动）

```
sparse_indices[B, Q_S, KV_N, sparse_size] → 取 sparse_size 个 block 索引
→ 计算物理 KV 偏移 → 间接 load KV（INT8/FP8）
```

### 3.2 KV 反量化（Per-Token-Head-Tile-128）

量化参数和数据混合存放（quant_scale_repo_mode=1）：
- K D 维 = nope(512) + rope(128) + dequant_scale(4×4) = 656
- 反量化：`K_fp32 = cast(K_int8/fp8, fp32) * dequant_scale[tile_id]`
- per-tile: tile_size=128，每个 tile 共享一个 scale

### 3.3 MLA-absorb 模式（attention_mode=2）

Q/K 沿 D 维拼接 nope+rope（K 额外含 dequant_scale），分段 dot：
```
s = q_nope · k_nope^T + q_rope · k_rope^T
```

### 3.4 在线 FlashSoftmax + sparse mask

sparse_mode=3 rightDownCausal mask 后，FlashSoftmax 在线刷新，FP32 累加。

## 4. 模板与 dispatch

| 参数 | 取值 | 说明 |
|------|------|------|
| FLASH_DECODE | 0/1 | FlashDecode 使能 |
| LAYOUT_T | 0:BSND / 1:TND | query layout |
| KV_LAYOUT_T | 0:BSND / 1:TND / 2:PA_BSND | KV layout |
| TEMPLATE_MODE | 0:C_TEMPLATE / 1:V_TEMPLATE | Cube/Vector 模板 |

5 种合法组合（`template_tiling_key.h` L42-55）。

## 5. 架构与产品支持

| 产品 | 架构 | config（def.cpp） | kernel 入口 | KV dtype |
|------|------|-------------------|-------------|----------|
| Atlas A2 推理 | arch32 | `ascend910b` L92 | `kv_quant_sparse_flash_attention.cpp` | INT8 |
| Atlas A3 推理 | ascend910_93 | `ascend910_93` L93 | 同上 | INT8 |
| Ascend 950PR/950DT | arch35 | `ascend950` L151 | 同上（arch35 分支） | FP8_E4M3FN/HIFLOAT8 |

> 仅推理场景，无训练，无 310P。

## 6. Agent actions

1. 确认目标架构（arch32 INT8 / arch35 FP8）与 layout（BSND/TND/PA_BSND）。
2. sparse_indices 驱动离散取数：`tl.load(indices)` → 间接 `tl.load(KV)`。
3. KV per-tile-128 反量化：`cast(int8/fp8, fp32) * dequant_scale[tile_id]`。
4. MLA-absorb：Q/K 沿 D 拼接，分段 dot（nope + rope）。
5. 参考 [[technique-sparse-kv-gather]] 了解搬运聚合优化，[[technique-sparse-attention-tiling]] 了解 SFA tiling 策略，[[technique-sparse-attention-pipeline]] 了解流水编排。
6. Triton 实现参考 `artifacts/kv-quant-sparse-flash-attention/kv-quant-sparse-flash-attention-triton-skeleton.py`（pseudocode）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、per-tile 量化精度边界、sparse_indices 有效值约定、FP8_E4M3FN/HIFLOAT8 精度边界需结合源码/硬件进一步验证。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 [[ops-transformer-kv-quant-sparse-flash-attention]]。