---
id: kernel-block-sparse-attention
title: BlockSparseAttention (block-sparse mask) on Ascend NPU
type: kernel
tags: [block_sparse_attention, sparse_attention, flash_attention, online_softmax, cube_vector_pipe, mask_compression, kv_gather, gqa]
techniques: [sparse_attention_tiling, sparse_kv_gather, fp32_accumulation, tail_mask]
hardware_features: [unified_buffer, l1_buffer, l0a_buffer, l0b_buffer, l0c_buffer, cube_vector_1_to_2, mte2, mte3, buffer_id_sync]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [attention]
operators: [block_sparse_attention]
symptoms: [performance_bottleneck, precision_mismatch]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/block-sparse-attention
related: [technique-block-sparse-mask-gather, technique-flash-attention-pipeline, kernel-sparse-flash-attention, kernel-flash-attention-score, kernel-prompt-flash-attention]
sources: [ops-transformer-block-sparse-attention]
performance_claims: []
---

# BlockSparseAttention（块稀疏注意力）

CANN ops-transformer 的 `BlockSparseAttention`（内部代号 **RFA = Rain Fusion Attention**）是面向**灵活块级稀疏模式**的注意力算子：通过 `blockSparseMask`（INT8，`[batch, headNum, ceilDiv(Q,blockShapeX), ceilDiv(KV,blockShapeY)]`）指定每个 Q 块选择哪些 KV 块，只对选中块计算，实现真正的计算稀疏（docs L16-22）。它与 [[kernel-flash-attention-score]]、[[kernel-sparse-flash-attention]] 共享 QK->在线 softmax->PV 骨架（[[technique-flash-attention-pipeline]]），核心差异是 **blockSparseMask 驱动的块级稀疏**：mask 被压缩为 CSR 式索引，GEMM 仅 gather 加载选中 KV 块（[[technique-block-sparse-mask-gather]]）。区别于 SFA 用外部 `sparse_indices` 离散取数，本算子用 4D 掩码描述块级 pattern。基于类 CUTLASS 的 `attn_infra` 模板库实现。来源见 [[ops-transformer-block-sparse-attention]]。

## 1. 输入输出

### 输入（def.cpp L19-54；docs L77-236）

| 输入 | 必选/可选 | 含义 | dtype | shape 约束 |
|---|---|---|---|---|
| query | 必选 | Q | FP16/BF16 | TND `[T,N1,D]` / BNSD `[B,N1,S,D]`；D∈{64,128} |
| key | 必选 | K | FP16/BF16 | TND `[T,N2,D]` / BNSD `[B,N2,S,D]` |
| value | 必选 | V | FP16/BF16 | 同 key |
| blockSparseMask | 必选 | 块稀疏 pattern（1=算/0=跳） | INT8 | `[B, N1, ceilDiv(maxQS, blockShapeX), ceilDiv(maxKvS, blockShapeY)]` |
| attenMask | 必传 nullptr | - | - | 当前不支持 |
| blockShape | 可选 | `[blockShapeX, blockShapeY]` | INT64 | X>0，Y>0 且 128 倍数；默认 `[128,128]` |
| actualSeqLengths / actualSeqLengthsKv | TND 必选 | 每 batch Q/KV 实际长 | INT64 | 须同时配/同时不配 |
| blockTable | 必传 nullptr | - | - | PagedAttention 不支持 |

### 输出（def.cpp L55-62；infershape.cpp L105,183-201）

| 输出 | 含义 | dtype | shape |
|---|---|---|---|
| attentionOut | 注意力输出 | FP16/BF16（同 query） | 同 query |
| softmaxLse | log-sum-exp（softmaxLseFlag=1 时） | FLOAT | TND `[T,N1,1]` / BNSD `[B,N1,S,1]` |

### 属性（def.cpp L64-73）

- `qInputLayout`/`kvInputLayout`（默认 TND；TND/BNSD，须一致）、`numKeyValueHeads`（默认1）。
- `maskType`（仅 0）、`scaleValue`（默认 0，实际取 1/√D，[needs_verification] null 语义）、`innerPrecise`（0=fp32 softmax，1=fp16 softmax；BF16 仅 0；4=LOW_HIGH_MIXED 无 kernel 路径 [needs_verification]）。
- `blockSize`（仅 0）、`preTokens`/`nextTokens`（仅 INT_MAX，不支持滑窗）、`softmaxLseFlag`（0/1）。

### 约束（docs L487-509；README L156-166）
- q/k/v dtype 一致（FP16/BF16），layout 一致（TND-TND 或 BNSD-BNSD），D∈{64,128}。
- blockShapeY 须 128 倍数；qSeqlen/kvSeqlen 不必被 blockShape 整除（ceilDiv）。
- GQA：N1 ≥ N2 且 N1%N2==0。
- 无 PagedAttention、无滑窗、无 causal mask（maskType 仅 0）。

## 2. 核心计算公式

```
attentionOut = Softmax(scale · query · key_sparse^T + atten_mask) · value_sparse
```
- `key_sparse`/`value_sparse` 为 `blockSparseMask` 选中的 KV 块（docs L18-22）。
- `scale` 默认 D^-0.5。
- 在线 FlashSoftmax：running row-max `m_i` / row-sum `l_i`，跨 KV 块 `alpha=exp(m_old-m_new)` rescale + PV 累加（同 FA）。
- 全 0 mask 行（`curSelectNum==0`）整 Q-block 跳过，不产生 GEMM 工作（kernel.h L510-514）。

## 3. 计算流程（4 阶段 cube↔vector 流水 + mask 压缩）

骨架同 [[technique-flash-attention-pipeline]]；块稀疏 mask 压缩是本算子独有前置，详见 [[technique-block-sparse-mask-gather]]。

**Stage0 mask 压缩（AIV，全局一次）**：`Mask2IdxAndCount`（kernel.h L93-248）把 INT8 mask 压缩为 `selectIdx`（选中 KV-block 索引，int32）+ `selectNumIdx`（每行选中数，int32）。`SyncAll` 屏障后供 cube 使用。

**Stage1-4 流水（3 路循环 buffer，`PRE_LAUNCH=2`）**：

| 阶段 | 单元 | 入口 | 操作 |
|---|---|---|---|
| Stage1 QK(bmm1) | AIC | kernel.h L616-633 | Q·K^T（沿 selectIdx gather K）-> set qkReady |
| Stage2 在线 softmax | AIV | kernel.h L642-652 | wait qkReady；max/sum + rescale -> set softmaxReady |
| Stage3 PV(bmm2) | AIC | kernel.h L678-697 | wait softmaxReady；P·V（沿 selectIdx gather V）-> set pvReady |
| Stage4 rescale O/写回 | AIV | kernel.h L717-731 | wait pvReady；O/=l_i；可选写 LSE |

3 个 cross-core flag（qkReady/softmaxReady/pvReady）为 CANN 专有，Triton 侧抽象为 tile 依赖。FP32（innerPrecise=0）/ FP16（=1）softmax 由 `block_epilogue_online_softmax[_low_prec]` 两个特化实现。

## 4. tiling 与分核（概要，详见 artifact tiling_dispatch_tables.md）

- 12 个 tiling key（`tilingkey.h` L31-65），按 **dtype × Q/KV-layout × softmax-precision × softmaxLseFlag** 分派；D(64/128) 与 blockShape 不进 key。base `RFA_BASE_TILING=9000000000000000`。
- 核间仅在 **(Q-block × head)** 轴切分：`blockDim=min(aicNum, totalTaskNum)`，`totalTaskNum=Σ_b ceilDiv(qseqlen_b, blockShapeX)·numHeads`（tiling.cpp L557-603）。**无 FlashDecoding、无 S2 核间切分**——KV 在核内沿 selectIdx 稀疏遍历。
- GQA：`groupSize=numHeads/numKvHeads`，每 KV head 被 groupSize 个 Q head 共享。
- 基本块 `BASIC_BLOCK_SIZE=128`、`Q_TILE_CEIL=128`；tile shape QK `<128,128,128>`、PV L1 `<128,128,256>`（interface.cpp L68-79）。
- 非对齐 ceilDiv + GEMM 内 tail（`block_mmad_qk.hpp` L200-201）。
- workspace 三缓冲：`[gS|gP|gOTmp|gOUpdate|gSelectNumIdx|gSelectIdx]` + sync buffer（tiling.cpp L605-630）。
- 任务类型 `KERNEL_TYPE_MIX_AIC_1_2`（AIC:AIV=1:2）。

## 5. 变体对照

| 变体 | 算子 | 场景 | 稀疏机制 |
|---|---|---|---|
| 训练前向 | flash_attention_score | 训练，连续 KV | 无（dense）/ atten_mask |
| 推理 prefill | prompt_flash_attention | 推理全序列 | 无 / atten_mask |
| **稀疏推理（索引）** | sparse_flash_attention | 大序列推理，MLA-absorb | `sparse_indices` 离散 KV gather |
| **块稀疏** | **block_sparse_attention** | 块级稀疏 pattern | `blockSparseMask` 4D 掩码 -> CSR 压缩 -> gather |
| 增量解码 | incre_flash_attention | KV cache 增量 | - |

相对 FA/SFA 的核心差异：blockSparseMask 块级掩码驱动（非 index 驱动）、mask 压缩为 CSR 后 gather 仅算选中块、无 MLA、无 FlashDecoding、类 CUTLASS `attn_infra` 实现。

## 6. 架构支持（docs L5-12；def.cpp L75-76）

| 产品 | 架构 | 支持 |
|---|---|---|
| Atlas A2 训练/800I A2 推理/A200I A2 Box | Ascend910B1 | √ |
| Atlas A3 训练/推理系列 | Ascend910_93 | √ |

不支持 950PR/950DT/310P/训练系列。仅 FP16/BF16。`attn_infra/arch/arch.hpp` 仅实现 AtlasA2（C220）。

## 7. Agent actions

1. 按 `qInputLayout`/`kvInputLayout`（TND/BNSD，须一致）解析 Q/KV 地址；D∈{64,128}。
2. **块稀疏 mask 压缩**：把 `blockSparseMask`（INT8）预处理为 `selectIdx`/`selectNumIdx`（CANN 在 vector 核全局做一次；Triton 侧 host 预处理或独立 kernel）。参考 [[technique-block-sparse-mask-gather]]。
3. 稀疏 KV 遍历：沿 `selectIdx` gather K/V，仅计算选中块；`curSelectNum==0` 整 Q-block 跳过。
4. 在线 FlashSoftmax：running max/sum + `exp(m_old-m_new)` rescale + PV 累加（同 FA）。`innerPrecise=0` 用 FP32，`=1` 用 FP16（仅 FP16 输入，注意溢出）。
5. GQA：`groupSize=numHeads/numKvHeads`，KV head 跨 Q head 共享。
6. 核间切分：program 维 = (batch, head, Q-block)；KV 不切核。`BLOCK_M=128`（source-reported）。
7. 可选输出 `softmaxLse`（softmaxLseFlag=1）。
8. Triton 实现参考 `artifacts/block-sparse-attention/block-sparse-attention-triton-skeleton.py`（pseudocode）。CANN 专有机制（MTE/CrossCore flag/PreloadPipeline/TQue/ND↔NZ）不迁移。

## 8. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、UB/L1 占用、mask 压缩并行策略、innerPrecise 精度边界、scaleValue null 默认语义均未经硬件实测；`maskType=3`/`innerPrecise=4`/`PAGED_CACHE_FLAG` 等死代码或未实现路径见 artifact migration-table.md 的 `needs verification` 项。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 [[ops-transformer-block-sparse-attention]]。
