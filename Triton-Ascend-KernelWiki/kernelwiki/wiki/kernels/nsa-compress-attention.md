---
id: kernel-nsa-compress-attention
title: NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU
type: kernel
tags: [nsa, native_sparse_attention, attention, compress_attention, topk, sliding_window, importance_score]
techniques: [fp32_accumulation, tail_mask]
hardware_features: [unified_buffer, l0c_buffer, l1_buffer, l0c_to_ub, cube_vector_1_to_2]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [attention]
operators: [nsa_compress_attention]
symptoms: [performance_bottleneck, ub_overflow]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/nsa-compress-attention
related: [kernel-flash-attention-score, technique-flash-attention-tiling, technique-flash-attention-pipeline]
sources: [ops-transformer-nsa-compress-attention]
performance_claims: []
---

# NsaCompressAttention (NSA 压缩注意力 + topk 索引)

CANN ops-transformer 中的 `NsaCompressAttention` 算子实现 NSA（Native Sparse Attention，arxiv 2502.11089）的压缩注意力阶段，包括压缩注意力计算、importance score 滑动窗口聚合、以及 topk 索引选择。输出供下游 `nsa_selected_attention` 算子使用。

仅支持 Atlas A2 训练系列（arch32 / ascend910b / ascend910_93），不支持 ascend950。仅支持 TND layout（varlen）。与 `flash_attention_score` 相似但更简单：仅 2 次 matmul（Q·K^T + P·V），S2 不做切分。

## 1. 输入输出

### 输入（按 op_host/nsa_compress_attention_def.cpp 顺序）

| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| query | 必选 | 查询 Q，TND 格式 (T1, N1(=G), D) | FP16/BF16 |
| key | 必选 | 键 K，TND 格式 (T2, N2, D) | FP16/BF16 |
| value | 必选 | 值 V，TND 格式 (T2, N2, Dv) | FP16/BF16 |
| atten_mask | 可选 | 注意力掩码，1=不参与 0=参与 | BOOL/UINT8 |
| actual_seq_qlen | 可选 | query 变长序列长度（前缀和） | INT64 |
| actual_cmp_seq_kvlen | 可选 | kv 压缩序列长度（前缀和） | INT64 |
| actual_sel_seq_kvlen | 可选 | kv 选择序列长度（前缀和，`P_slc'`第0维） | INT64 |
| topk_mask | 可选 | topk 掩码 | BOOL/UINT8 |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| softmax_max | Softmax 行最大值，供反向 | FLOAT |
| softmax_sum | Softmax 行求和，供反向 | FLOAT |
| attention_out | 注意力输出，与 query 形状一致 | FP16/BF16 |
| topk_indices_out | topk 索引 | INT32 |

### 维度约定（README.md L42-47）

TND 格式：T=BS 合轴紧密排列，N=head_num，D=head_dim。`N1 >= N2 && N1 % N2 == 0`，`G = N1 / N2`，`G < 128 && 128 % G == 0`。`qD == kD && kD >= vD`。D ≤ 768。

### 属性（def.cpp L92-99）

`scale_value`(1.0)、`head_num`(必需)、`input_layout`(必需，仅TND)、`sparse_mode`(0)、`compress_block_size`(l, 必需)、`compress_stride`(d, 必需)、`select_block_size`(l', 必需)、`select_block_count`(n, 必需)。

## 2. 核心公式（README.md L16-38）

| 步 | 公式 | 说明 |
|----|------|------|
| 1 | `P_cmp = Softmax(Q * K^T)` | 压缩注意力（无 scale） |
| 2 | `attention_out = Softmax(mask(scale * Q * K^T, atten_mask)) * V` | 注意力输出（含 mask + scale） |
| 3 | `P_slc[j] = sum_{m=0}^{l'/d-1} sum_{n=0}^{l/d-1} P_cmp[l'/d * j - m - n]` | importance score 滑动窗口聚合 |
| 4 | `P_slc' = sum_{h=1}^{H} P_slc^h` | 跨 head 累加 |
| 5 | `P_slc' = topk_mask(P_slc')` | 可选 topk 掩码 |
| 6 | `topkIndices = topk(P_slc')` | topk 索引输出 |

常量：`isM = l' / d`（select block 内压缩步数），`isN = l / d`（compress block 内压缩步数），`S2PerScore = isM + isN - 1`，`SelSkv = CeilDiv(CmpSkv, isM)`。

## 3. 计算流程（2 阶段 Cube + 2 阶段 Vector）

| 阶段 | 入口函数 | 单元 | 对应步 |
|------|----------|------|--------|
| Cube1 | IterateBmm1 | AIC | 步 1-2（Q·K^T，mm1） |
| Vector1 | ProcessVec1 | AIV | 步 2（scale → mask → Softmax → 写回 workspace） |
| Cube2 | IterateBmm2 | AIC | 步 2（P·V，mm2 → attention_out） |
| Vector2 | ProcessVec2 + TopkCompute | AIV | 步 3-6（importance score 滑动窗口 + 跨 head 累加 + topk_mask + topk） |

## 4. Tiling 与核切分

- `s1BasicBlock = NSA_BASE_S1G_SIZE / gSize = 128 / gSize`（`tiling_general.cpp` L873）
- `s1VecBasicBlock = s1BasicBlock / 2`（CV_RATIO=2，L32,874）
- `s2BasicBlock = alignedS2`（S2 不做切分，整块计算，L875）
- 核间切分：`totalSize = accumS1BlockNum * n2Size`，`coreNum = min(totalSize, aicNum) * 2`（SameAB AIC×2，L924-942）
- workspace：`pWorkspaceSize + impScoreWorkspaceSize + stage1ResWorkspaceSize + 16MB`（L949-958）
- UB 分配：`availUbSize = 192KB`（kernel L315），Softmax 区域 + importance score 区域 + topk 区域（L316-379）

## 5. 模板 dispatch（kernel L74-112）

- **1 个 kernel 模板**：`NsaCompressAttentionS1s2Bn2gs1SameAB`（`nsa_compress_attention_s1s2_bn2gs1_sab.h`）
- 1 个 varlen 变体：`NsaCompressAttentionVarLenScoreSameAB`（`nsa_compress_attention_var_len_score_sab.h`）
- 8 分支 dispatch：tiling key 固定 `10000000000000000024UL`，按 dtype（FP16/BF16）× hasAttenMask × hasTopkMask 区分
- 不支持 ascend950（仅 ascend910b/910_93，def.cpp L112-113）

## 6. 约束（README.md L67-85）

- compressBlockSize(l): 16~128，16对齐；compressStride(d): 16~64，16对齐；l ≥ d
- selectBlockSize(l'): 16~128，16对齐；l' ≥ l，l' % d == 0
- selectBlockCount(n): [1, 32]，≤ min(SelSkv)
- `CmpSkv ≤ 14000`（UB 限制）
- layout 仅支持 TND；query/key/value dtype 一致；`G < 128 && 128 % G == 0`

## 7. Agent actions

1. 确认 compressBlockSize/compressStride/selectBlockSize/selectBlockCount 参数，计算 isM/isN。
2. 按 TND layout 解析 query/key/value 的 T、N、D 维度，计算 G = N1/N2。
3. 设置 `s1BasicBlock = 128 / gSize`，`s2BasicBlock = alignedS2`，S2 不做切分。
4. 设置 `BLOCK_M` = s1BasicBlock，`BLOCK_N` = Skv（整块）。
5. 编排 2 次 matmul（Q·K^T + P·V）和 importance score 滑动窗口 + topk。
6. 检查 atten_mask/topk_mask 是否存在，选对应 constexpr 分支。
7. Triton 实现参考 `artifacts/nsa-compress-attention/nsa-compress-attention-triton-skeleton.py`（pseudocode）。

## 8. 复现性声明

本页结论基于源码静态分析（confidence: inferred），tiling 参数数值、UB 占用、topk UB 限制未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 [[ops-transformer-nsa-compress-attention]]。