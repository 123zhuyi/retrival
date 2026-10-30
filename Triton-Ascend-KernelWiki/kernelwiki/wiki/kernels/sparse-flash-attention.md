---
id: kernel-sparse-flash-attention
title: SparseFlashAttention inference (sparse KV) on Ascend NPU
type: kernel
tags: [flash_attention, sparse_attention, softmax, online_softmax, kv_cache, page_attention, mla, cube_vector_pipe, sparse_kv_gather]
techniques: [sparse_attention_tiling, sparse_attention_pipeline, sparse_kv_gather, fp32_accumulation, tail_mask]
hardware_features: [unified_buffer, l0c_to_ub, cube_vector_1_to_2, l1_buffer, l0c_buffer]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [attention]
operators: [sparse_flash_attention]
symptoms: [performance_bottleneck, precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/sparse-flash-attention
related: [kernel-flash-attention-score, kernel-prompt-flash-attention, technique-sparse-attention-tiling, technique-sparse-attention-pipeline, technique-sparse-kv-gather]
sources: [ops-transformer-sparse-flash-attention]
performance_claims: []
---

# SparseFlashAttention（稀疏注意力推理）

CANN ops-transformer 的 `SparseFlashAttention`（SFA）是面向**大序列推理场景**的稀疏注意力算子：通过外部选择算法（如 `lightning_indexer`）得到重要的稀疏 K/V，"只计算关键部分"大幅减少计算量，并针对离散访存做指令缩减与搬运聚合优化（README L12-21）。它与训练前向 [[kernel-flash-attention-score]]、推理 prefill [[kernel-prompt-flash-attention]] 共享 QK->FlashSoftmax->PV 骨架，核心差异是 **`sparse_indices` 驱动的离散 KV 取数**与 **MLA-absorb**。可复用的 tiling/流水/离散 KV 搬运知识分别见 [[technique-sparse-attention-tiling]]、[[technique-sparse-attention-pipeline]]、[[technique-sparse-kv-gather]]。

## 1. 输入输出

### 输入（def.cpp L23-67；README L36-55）

| 输入 | 必选/可选 | 含义 | dtype | shape 约束 |
|---|---|---|---|---|
| query | 必选 | Q | FP16/BF16 | [B,S1,N1,D] (BSND) / [T1,N1,D] (TND)；N1∈{1,2,4,8,16,32,64,128}；D=512 |
| key | 必选 | K̃（稀疏） | FP16/BF16 | [B,S2,KV_N,D] / [T2,KV_N,D] / [block_num,block_size,KV_N,D] (PA)；KV_N=1 |
| value | 必选 | Ṽ | FP16/BF16 | 同 key |
| sparse_indices | 必选 | 离散取 kvCache 的索引 | INT32 | [B,Q_S,KV_N,sparse_size]；有效值在前、无效在后 |
| block_table | 可选 | PageAttention 块表 | INT32 | [B, ≥ceil(S2_max/block_size)] |
| actual_seq_lengths_query / _kv | 可选 | 变长前缀和 | INT32 | TND/PA 时必传 |
| query_rope / key_rope | 可选 | MLA rope | FP16/BF16 | D=64 |

### 输出（def.cpp L68-79；README L79-81）

| 输出 | 含义 | dtype | shape |
|---|---|---|---|
| attention_out | 注意力输出 | FP16/BF16 | [B,S1,N1,D] / [T1,N1,D] |
| softmax_max | 行 max（return_softmax_lse=True 时） | FLOAT | [B,N2,S1,N1/N2] / [N2,T1,N1/N2] |
| softmax_sum | 行 sum（return_softmax_lse=True 时） | FLOAT | 同 softmax_max |

注：`return_softmax_lse` 默认 False；仅训练且 `layout_kv` 非 PA_BSND 场景支持返回 max/sum（README L75）。

### 属性（def.cpp L80-88）

- `scale_value`（必选，默认1.0）、`sparse_block_size`（默认1，[1,128] 且 2 的幂；1=token-wise，>1=block-wise）。
- `layout_query`（默认BSND：BSND/TND）、`layout_kv`（默认BSND：TND/BSND/PA_BSND）。
- `sparse_mode`（默认3=rightDownCausal 下三角；0=全部）。
- `pre_tokens`/`next_tokens`（仅默认 INT64_MAX）、`attention_mode`（仅 2=MLA-absorb）、`return_softmax_lse`（默认false）。

### 约束（README L83-90）
- 仅推理场景，支持图模式。
- D(nope)=512，D(rope)=64（MLA 合计 576）；Q/K/V dtype 一致。
- `sparse_block_size` 整除 `block_size`；`block_size` 为 16 倍数、最大 1024。
- `layout_query=TND` + `layout_kv=BSND` 不支持。

## 2. 核心计算公式

```
attention_out = Softmax(scale · (Q · K̃^T)) · Ṽ
```
- K̃/Ṽ 为 sparse_indices 选取的重要 K/V（README L16-20）。
- **MLA-absorb**（attention_mode=2）：Q/K 的 nope(512) 与 rope(64) 沿 D 维拼接成 576 用于 QK；key/value 共享同一底层张量（README L73）。
- sparse_mode=3：rightDownCausal 下三角 mask。
- 无效 sparse_indices（-1）跳过/遮蔽。

## 3. 计算流程（5 阶段 CV 流水，V_TEMPLATE）

详见 [[technique-sparse-attention-pipeline]]。V_TEMPLATE 下比标准 FA 多一个 MergeKv 前置阶段：

| 阶段 | 单元 | 入口 | 操作 |
|---|---|---|---|
| Stage0 MergeKv | AIV | `service_vector_mla.h:986` | 按 sparse_indices 从 GM 离散读 KV，聚合到连续 buffer |
| Stage1 Bmm1 | AIC | `kernel_mla.h:729` | Q·K^T |
| Stage2 Vec1 | AIV | `service_vector_mla.h` | FlashSoftmax（在线 max/sum） |
| Stage3 Bmm2 | AIC | `kernel_mla.h:743` | P·V |
| Stage4 Vec2 | AIV | `service_vector_mla.h` | rescale + 写回 |

C_TEMPLATE 无 MergeKv（Cube 直接处理稀疏 block）。在线 FlashSoftmax 同 FA：`exp[i]=e^(max[i-1]-max[i])`，PV rescale 累加。

## 4. tiling 与分核（概要，详见 [[technique-sparse-attention-tiling]]）

- 2 模板（C_TEMPLATE/V_TEMPLATE），由 `sparseBlockSize≤4 && s2Size≠0` 触发 V_TEMPLATE。
- 基本块：`mBaseSize`=128（S1）、`s2BaseSize` 动态、`PRELOAD_NUM`=2、3-slot 循环 buffer。
- 核间切分：B/N2/G/S1 四轴负载均衡（`ProcessBalance`）；FlashDecoding 时 S2 也可核间切分（与标准 FA 不同，稀疏 block softmax 可独立合并）。
- MLA：D=576，headDimAlign=Align(qkHeadDim,32B)。
- 任务类型 `KERNEL_TYPE_MIX_AIC_1_2`（AIC:AIV=1:2，kernel 入口 L41）。
- 5 个 CV sync event（syncV0C1/syncC1V1/syncV1C2/syncC2V2/syncC2V1，CANN 专有）。

## 5. 变体对照

| 变体 | 算子 | 场景 |
|---|---|---|
| 训练前向 | flash_attention_score | 训练，连续 KV |
| 推理 prefill | prompt_flash_attention | 推理全序列，连续 KV，输出 lse |
| **稀疏推理** | **sparse_flash_attention** | 大序列推理，sparse_indices 驱动离散 KV，MLA-absorb |
| 增量解码 | incre_flash_attention | KV cache 增量 |

SFA 相对 FA/PFA 的核心差异：sparse_indices 驱动的离散 S2 遍历、MergeKv 离散->连续聚合、MLA-absorb（D=576）、FlashDecoding S2 核间切分、仅 A2/A3 推理。

## 6. 架构支持（README L5-8；def.cpp L96-97）

| 产品 | 架构 | 支持 |
|---|---|---|
| Atlas A2 推理 | Ascend910B1 | √ |
| Atlas A3 推理 | Ascend910_93 | √ |

不支持 950/310P/训练系列。仅 FP16/BF16。

## 7. Agent actions

1. 确认 `attention_mode=2`（MLA-absorb）：Q/K 沿 D 拼 nope(512)+rope(64)=576，key/value 共享张量。
2. 按 `sparseBlockSize` 选 C_TEMPLATE（>4）或 V_TEMPLATE（≤4，含 MergeKv）。
3. S2 循环由 `sparse_indices` 驱动：读 block 索引 -> 跳过无效(-1) -> 聚合为连续 chunk -> QK/PV。
4. PagedAttention 场景：block_table 查物理 block 再取 KV。
5. FlashDecoding 场景：S2 核间切分，各核局部 max/sum/PV 后跨核 reduce 合并。
6. 参考 [[technique-sparse-kv-gather]] 的离散 KV 搬运聚合、[[technique-sparse-attention-pipeline]] 的 5 阶段流水、[[technique-sparse-attention-tiling]] 的分核。
7. Triton 实现参考 `artifacts/sparse-flash-attention/sparse-flash-attention-triton-skeleton.py`（pseudocode）。

## 8. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、UB 占用、FlashDecoding 合并数值稳定性、PA offset 算术均未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 [[ops-transformer-sparse-flash-attention]]。
