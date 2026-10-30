---
id: kernel-fused-infer-attention-score
title: FusedInferAttentionScore inference (prefill+decode) on Ascend NPU
type: kernel
tags: [fused_infer_attention_score, attention, flash_attention, online_softmax, flash_decode, kv_cache, paged_attention, attn_infra]
techniques: [fp32_accumulation, tail_mask]
hardware_features: [unified_buffer, l0c_buffer, l1_buffer, l0a_buffer, l0b_buffer, cube_vector_1_to_2]
architectures: [Ascend910B1, Ascend910B2, Ascend910_93, Ascend950PR, Ascend950DT]
kernel_types: [attention]
operators: [fused_infer_attention_score]
symptoms: [precision_mismatch, ub_overflow, performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/fused-infer-attention-score
related: [technique-flash-attention-tiling, technique-flash-attention-pipeline, pattern-flash-attention-tiling-key-dispatch, kernel-flash-attention-score, kernel-prompt-flash-attention, kernel-incre-flash-attention]
sources: [ops-transformer-fused-infer-attention-score]
performance_claims: []
---

# FusedInferAttentionScore (推理统一 prefill + decode)

CANN ops-transformer 中的 `FusedInferAttentionScore`（FIA）算子是**推理统一 attention**：既支持 prefill（Q_S>1，对应 PromptFlashAttention）也支持 decode（Q_S=1，对应 IncreFlashAttention）。架构覆盖 Atlas A2（arch32）/ Atlas A3 训练（ascend910_93）/ Ascend 950PR/950DT（arch35），不支持 310P。核心架构特色是 op_kernel 的 **`attn_infra/` 分层模板库**（gemm/epilogue/layout/arch）。5 个版本 V1-V5，V5 最新（仅 950，新增 qStartIdx/kvStartIdx/pseType，支持 FP8/FP4/INT4、分离 KV 反量化、sharedPrefix、RoPE 解耦、learnableSink、softmaxLse 输出）。本页归纳语义与流程，可复用 tiling/流水知识见 [[technique-flash-attention-tiling]]、[[technique-flash-attention-pipeline]]、`artifacts/fused-infer-attention-score/`。与 [[kernel-flash-attention-score]]、[[kernel-prompt-flash-attention]]、[[kernel-incre-flash-attention]] 共享 QK->softmax->PV 骨架，差异在统一双场景、attn_infra 分层架构与 V5 新特性。

## 1. 输入输出（V5 主语义）

### 输入（按 V5 aclnn L55-105 / def.cpp）

| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| query | 必选 | 查询（Q_S>1 prefill / Q_S=1 decode） | FP16/BF16/INT8/HIFLOAT8/FLOAT8_E4M3FN |
| key / value | 必选(tensorlist) | 键/值，KV Cache | FP16/BF16/INT8/HIFLOAT8/FLOAT8_E4M3FN/INT4(INT32)/FLOAT4_E2M1 |
| pse_shift | 可选 | 位置编码（pseType 控制顺序） | FP16/BF16 |
| atten_mask | 可选 | 注意力掩码 | BOOL/INT8/UINT8 |
| actual_seq_lengths / actual_seq_lengths_kv | 可选 | per-batch Q/KV 有效长度 | INT64 |
| deq_scale1 / quant_scale1 / deq_scale2 / quant_scale2 / quant_offset2 | 可选 | 全量化因子 | UINT64/FLOAT32/BF16 |
| block_table | 可选 | PageAttention block 映射 | INT32 |
| query_padding_size / kv_padding_size | 可选 | Q/KV 右对齐 padding | INT64 |
| key_antiquant_scale / key_antiquant_offset | 可选 | K 分离反量化（V5，替代 antiquant_scale/offset） | FP16/BF16/FP32/FLOAT8_E8M0 |
| value_antiquant_scale / value_antiquant_offset | 可选 | V 分离反量化（V5） | 同 key |
| key_shared_prefix / value_shared_prefix / actual_shared_prefix_len | 可选 | 共享前缀 KV | 同 key/value |
| query_rope / key_rope | 可选 | RoPE 解耦（nope+rope 拼接） | 同 query |
| key_antiquant_rope_scale | 可选 | RoPE 反量化 scale | - |
| dequant_scale_query | 可选 | Q 反量化 | - |
| learnable_sink | 可选 | 可学习 sink | - |
| q_start_idx / kv_start_idx | 可选（V5 新增） | Q/KV 序列起始偏移 | INT64 |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| attention_out | 注意力输出 | FP16/BF16/INT8 |
| softmax_lse | softmax LSE（softmaxLseFlag 控制，供下游） | FLOAT |

### 属性（V5）

`num_heads`（必需）、`scale_value`、`pre_tokens`/`next_tokens`（sparse 窗口）、`input_layout`（BSH/BSND/BNSD/BNSD_BSND）、`num_key_value_heads`（GQA）、`sparse_mode`、`inner_precise`、`block_size`（PA）、`antiquant_mode`、`softmax_lse_flag`、`key_antiquant_mode`/`value_antiquant_mode`（分离反量化模式）、`query_quant_mode`、`pse_type`（V5 新增）。

### 维度约定

- B≤65536，N≤256，D≤512；S≤20971520（20M，长序列超时风险见 README L100-154）。
- prefill Q_S>1：B/N/D 对齐约束（INT8 32B 对齐，FP16/BF16 16B 对齐，INT4 64B 对齐）。
- decode Q_S=1：query/key/value 全 INT8 不支持；INT4 伪量化支持 INT4 或 INT32 拼接。

## 2. 核心计算公式

```
attention_out = Softmax(scale * (Q·K^T), atten_mask) · V
softmax_lse = log(softmax_sum) + softmax_max   (可选输出)
```

采用 FlashSoftmax 在线刷新（对 Skv 轴切分），FP32 中间累加。RoPE 解耦时 Q/K 沿 D 拼接 nope+rope 分段 dot。

## 3. 计算流程（attn_infra gemm + epilogue 交替）

| 阶段 | attn_infra 模块 | 操作 |
|------|----------------|------|
| Cube1 (Bmm1) | `gemm/block/block_mmad_qk` | Q·K^T -> attention_score（FP32 累加） |
| Vector1 (Vec1) | `epilogue/tile_common` + `epilogue/block/block_epilogue_online_softmax` | pse -> ×scale -> atten_mask select -> FlashSoftmax（在线 max/sum） |
| Cube2 (Bmm2) | `gemm/block/block_mmad_pv` | P·V |
| Vector2 (Vec2) | `epilogue/block/block_epilogue_rescale_o` | rescale(÷softmax_sum) -> 可选 quant -> 写回；可选 softmaxLse |

### FlashSoftmax 在线刷新

- `exp[i] = e^(max[i-1] - max[i])`
- `acc[i] = acc[i-1]·exp[i] + P_i·V_i`
- `attention_out = acc / softmax_sum`

### FlashDecode（decode 场景）

decode（Q_S=1）+ BN2 不足时，S2 外切 SplitKv，各核独立 Attention，lse 加权规约：`scale[i]=exp(lse[i])/Σexp(lse[i])`，`res=Σ split_res[i]·scale[i]`。

## 4. attn_infra 分层架构（FIA 核心）

| 分层 | 目录 | 职责 | Triton 抽象 |
|------|------|------|------------|
| gemm | `attn_infra/gemm/block/` | `block_mmad_qk`=Bmm1，`block_mmad_pv`=Bmm2（含 `_decode` 变体） | `tl.dot` |
| gemm tile_common | `attn_infra/gemm/tile_common/` | Cube 搬运（GM->L1->L0A/L0B->L0C->GM） | load/store（编译器） |
| epilogue | `attn_infra/epilogue/block/` | `block_epilogue_online_softmax`=FlashSoftmax，`block_epilogue_rescale_o`=写回，`CombineScale`=量化合并 | running softmax + `o=acc/l_i` |
| epilogue tile_common | `attn_infra/epilogue/tile_common/` | Vector 搬运+elemwise（pse/scale/mask） | elemwise |
| layout | `attn_infra/layout/` | ND/NZ 布局 | Triton layout（不迁移） |
| arch | `attn_infra/arch/` | resource/buffer/cross_core_sync | 编译器管理（不迁移） |

详见 `artifacts/fused-infer-attention-score/pipeline-diagram.md` §1。

## 5. 模板与变体

### 5.1 prefill / decode 模板

| 模板 | 文件 | 场景 | S1 | FlashDecode |
|------|------|------|----|----|
| regular | `flash_attention_regular.h`（1059 行） | prefill 全量 | Q_S>1 | 否 |
| regular_decode | `flash_attention_regular_decode.h`（745 行） | decode 增量 | Q_S=1 | 是 |

### 5.2 版本对照

| 版本 | aclnn | 相对前一版本 | 架构 |
|------|-------|------------|------|
| V1 | `aclnnFusedInferAttentionScore` | 基础统一 prefill+decode | A2/A3/950 |
| V2 | `...V2` | 扩展 | A2/A3/950 |
| V3 | `...V3` | MLA 等 | A2/A3/950 |
| V4 | `...V4` | 跨架构主力 | A2/A3/950 |
| V5 | `...V5` | **qStartIdx/kvStartIdx/pseType；FP8/FP4；分离 KV 反量化；sharedPrefix；RoPE；learnableSink；softmaxLse** | **仅 950** |

## 6. 架构与产品支持

| 产品 | 架构 | config（def.cpp） | kernel 入口 | tiling |
|------|------|-------------------|-------------|--------|
| Atlas A2 训练/推理 | arch32（910B） | `ascend910b` L570 | `fused_infer_attention_score` | `tiling.cpp`+`arch32/tiling_v3.cpp` |
| Atlas A3 训练/推理 | ascend910_93 | `ascend910_93` L571 | arch32 | arch32 |
| Ascend 950PR/950DT | arch35 | `ascend950` L2050 | `fused_infer_attention_score_apt` | `arch35/tiling_v2.cpp` |

- 不支持 310P / Atlas 200I/500 A2 / Atlas 训练系列（A1）。
- V5 接口仅 950（V5 doc L10-15），V4 及之前跨 A2/A3/950；V5 在 A2/A3 支持范围 `needs_verification`。
- arch35 用 `attn_infra/` 分层 + `_template_tiling_key.h`（3116 行）`if constexpr` 分发；arch32 用 `tilingkey.h` 十进制 + regular/regular_decode 模板。

## 7. 特性机制

### 7.1 FlashDecode
decode + BN2 不足时 split-KV + lse 加权规约（同 IFA），可迁移为两段式 kernel。

### 7.2 分离 KV 反量化（V5）
`key_antiquant_scale`/`value_antiquant_scale` 替代 `antiquant_scale`/`antiquant_offset`（后者 V5 不支持），K/V 各自反量化，支持 per-tensor/channel/token/token+head/token-group + PA 模式。

### 7.3 sharedPrefix（V5）
`key_shared_prefix`/`value_shared_prefix` 独立前缀 KV，与主 KV 拼接参与 attention。

### 7.4 RoPE 解耦（V5）
`query_rope`/`key_rope` 独立输入，Q/K 沿 D 拼接 nope+rope 分段 dot，V 仅用 nope 部分。

### 7.5 learnableSink / softmaxLse / qStartIdx
- `learnable_sink`：sink 项加入 attention 分母。
- `softmax_lse_flag`：输出 softmaxLse 供下游。
- `q_start_idx`/`kv_start_idx`（V5 新增）：Q/KV 序列起始偏移。

### 7.6 PageAttention / 量化
`block_table` 驱动间接 load；deq/quant scale 全量化 + 分离 KV 反量化；FP8/FP4/INT4 支持精度边界 `needs_verification`。

## 8. Agent actions

1. 确认架构（arch32 / arch35）与版本（V1-V5；V5 仅 950），选 tiling 路径。
2. 确认场景：prefill（Q_S>1，regular 模板）或 decode（Q_S=1，regular_decode + FlashDecode）。
3. 评估 FlashDecode：decode + `BN2<0.4×核数` -> split-KV 两段式；否则单 kernel。
4. 检查 V5 新特性：`key/value_antiquant_scale`（分离反量化）、`shared_prefix`、`query/key_rope`、`learnable_sink`、`softmax_lse_flag`、`q/kv_start_idx`、`pse_type`。
5. attn_infra 分层映射：gemm->`tl.dot`，epilogue->online softmax + rescale，layout/arch 不迁移。
6. 参考 [[technique-flash-attention-pipeline]] 安排 Cube/Vector 流水；Triton 实现参考 `artifacts/fused-infer-attention-score/fused-infer-attention-score-triton-skeleton.py`（pseudocode，统一 prefill/decode + FlashDecode）。
7. MLA 全量化与伪量化 MSD DD 模板（IFA 设计介绍标注仅 FIA 调用）在 FIA 中覆盖，迁移时按 V5 分离反量化路径处理。

## 9. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。FIA 无 `算子设计介绍.md`，tiling/pipeline 知识来自 tiling 源码与 aclnn doc；tiling 参数数值、`tiling.cpp` 精确行号、arch35 模板枚举、attn_infra 精确类名/行号、V5 在 A2/A3 支持范围、FP8/FP4/INT4 精度边界均 `needs_verification`。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。FlashDecode/softmaxLse 精确公式以 `attn_infra/epilogue/block/block_epilogue_online_softmax.hpp` 为准。来源见 [[ops-transformer-fused-infer-attention-score]]。
