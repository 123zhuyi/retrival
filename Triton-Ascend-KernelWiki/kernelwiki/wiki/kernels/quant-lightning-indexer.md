---
id: kernel-quant-lightning-indexer
title: QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU
type: kernel
tags: [sparse_flash_attention, sfa, lightning_indexer, quantized_attention, int8, fp8, page_attention, topk, inference]
techniques: [fp32_accumulation, tail_mask]
hardware_features: [unified_buffer, l0c_buffer, l1_buffer, l0c_to_ub, cube_vector_1_to_2]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR]
kernel_types: [attention]
operators: [quant_lightning_indexer]
symptoms: [performance_bottleneck, precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/quant-lightning-indexer
related: [technique-sparse-attention-tiling, technique-sparse-attention-pipeline, technique-sparse-kv-gather]
sources: [ops-transformer-quant-lightning-indexer]
performance_claims: []
---

# QuantLightningIndexer (SFA 量化稀疏索引选择)

CANN ops-transformer 中的 `QuantLightningIndexer` 算子是推理场景下 SparseFlashAttention（SFA）的前处理算子，选出关键的稀疏 token 并对输入 query/key 进行量化实现存8算8。输出 `sparse_indices`（INT32）供下游 SFA 算子使用。

支持 Atlas A2 推理系列（arch32, int8 输入）和 Ascend 950PR/950DT（arch35, fp8 输入）。支持 BSND、TND、PA_BSND（PageAttention）三种 layout。

## 1. 输入输出

### 输入（按 op_host/quant_lightning_indexer_def.cpp 顺序）

| 输入 | 必选/可选 | 含义 | dtype（arch32/arch35） |
|------|-----------|------|----------------------|
| query | 必选 | 量化查询 Q | int8 / fp8_e4m3fn, hifloat8 |
| key | 必选 | 量化键 K | int8 / fp8_e4m3fn, hifloat8 |
| weights | 必选 | 加权系数 W | FP16 / BF16 |
| query_dequant_scale | 必选 | Q 反量化系数 Scale_Q | FP16 / FLOAT |
| key_dequant_scale | 必选 | K 反量化系数 Scale_K^T | FP16 / FLOAT |
| actual_seq_lengths_query | 可选 | query 有效 token 数 | INT32 |
| actual_seq_lengths_key | 可选 | key 有效 token 数 | INT32 |
| block_table | 可选 | PA 块映射表 | INT32 |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| sparse_indices | 稀疏索引，shape [B,S1,N2,k] 或 [T,N2,k] | INT32 |

### 维度约定（README.md L32-33）

- D=128 固定（HEAD_DIM_LIMIT=128, tiling.h L75）
- N1（query head num）: arch32 支持 64, arch950 支持 24/64
- N2（key head num）: 仅支持 1
- G = N1/N2，G_SIZE_LIMIT=64, G_SIZE_LIMIT_950=24
- S1（query seq）、S2（key seq）、sparse_count [1,2048]
- block_size（PA）: 16 倍数，≤1024

### 属性

`query_quant_mode`(0 per-token-head)、`key_quant_mode`(0)、`layout_query`(BSND/TND)、`layout_key`(BSND/TND/PA_BSND)、`sparse_count`(2048)、`sparse_mode`(0/3)、`pre_tokens`(默认 INT64_MAX)、`next_tokens`(默认 INT64_MAX)

## 2. 核心公式（README.md L16-20）

```
out = Top_k{ W @ [ ReLU((Scale_Q ⊙ Scale_K^T) ⊙ (Q^Quant @ K^Quant^T)) ] }
```

| 步 | 公式 | 说明 |
|----|------|------|
| 1 | `Q^Quant @ K^Quant^T` | 量化 QK 相关性（int8/fp8 → FP32 累加）|
| 2 | `(Scale_Q ⊙ Scale_K^T) ⊙` | 反量化系数逐元素乘 |
| 3 | `ReLU()` | 过滤负相关信号 |
| 4 | `W @` | 加权聚合（跨 G 轴）|
| 5 | `Top_k(..., sparse_count)` | 选取前 k 个稀疏索引 |

## 3. 计算流程

| 阶段 | 入口 | 单元 | 操作 |
|------|------|------|------|
| Preload | QLIPreload::Init | AIC+AIV | 输入加载、反量化系数准备 |
| Cube | QKCompute | AIC | Q·K^T matmul（反量化 FP32） |
| Vector | IndexCompute | AIV | Scale_Q⊙Scale_K^T → ReLU → W@加权 |
| Topk | TopK | AIV | topk 索引选择 |
| WriteBack | WriteBack | AIV | sparse_indices 写回 GM |

## 4. Tiling 与 dispatch

- **模板**：单模板 `QLIPreload`，6 模板参数（DT_Q/DT_K/DT_OUT/PAGE_ATTENTION/Q_LAYOUT_T/K_LAYOUT_T）
- **arch32**：2 组合（int8 × PA/BSND/TND）
- **arch35**：6 组合（fp8_e4m3fn/hifloat8 × PA/BSND/TND）
- D=128 固定，S2 不切分（整块计算）
- `KERNEL_TYPE_MIX_AIC_1_2`（1 Cube:2 Vector）
- PageAttention：block_table 映射离散 KV 块

## 5. Agent actions

1. 确认架构（arch32 int8 vs arch35 fp8）与 layout（BSND/TND/PA_BSND）。
2. 加载反量化系数 Scale_Q/Scale_K^T，int8/fp8 输入在 FP32 计算。
3. 按 `Q·K^T → ReLU → W@ → topk` 编排计算。
4. sparse_mode=3 时应用因果掩码（rightDownCausal）。
5. PA_BSND 场景通过 block_table 索引离散 KV 块。
6. Triton 实现参考 `artifacts/quant-lightning-indexer/quant-lightning-indexer-triton-skeleton.py`（pseudocode）。

## 6. 复现性声明

本页结论基于源码静态分析（confidence: inferred），tiling 参数数值、UB 占用、PageAttention 运行时契约未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 [[ops-transformer-quant-lightning-indexer]]。