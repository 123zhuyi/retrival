---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: PromptFlashAttention inference/prefill on Ascend NPU
description: CANN ops-transformer 的 PromptFlashAttention（PFA）是全量推理场景的 FlashAttention 融合算子，支持 sparse 优化、actualSeqLengthsKv 变长优化、INT8/FP8 量化、高精度/高性能模式选择（README L16）。它与训练前向 FlashAttentionScore（见 kernel-flash-attention-score）共享 QK->FlashSoftmax->PV 的核心骨架...
tags: [flash_attention, attention, softmax, online_softmax, prompt_attention, prefill, kv_cache, paged_attention, quant, mla, cube_vector_pipe, fp32_accumulation, tail_mask, unified_buffer, l0c_to_ub, cube_vector_1_to_2, l1_buffer, l0c_buffer, ascend910b1, ascend910_93, ascend950pr, prompt_flash_attention]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/prompt_flash_attention/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/prompt_flash_attention/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/prompt-flash-attention.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/prompt_flash_attention/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/prompt_flash_attention/)。

# PromptFlashAttention（推理 / prefill）

CANN ops-transformer 的 `PromptFlashAttention`（PFA）是**全量推理场景**的 FlashAttention 融合算子，支持 sparse 优化、actualSeqLengthsKv 变长优化、INT8/FP8 量化、高精度/高性能模式选择（README L16）。它与训练前向 `FlashAttentionScore`（见 `kernel-flash-attention-score`）共享 QK->FlashSoftmax->PV 的核心骨架，差异在于推理语义、softmaxLse 输出、KV-cache/PagedAttention/量化/MLA 扩展。可复用的 tiling/流水/dispatch 知识分别见 `technique-flash-attention-tiling`、`technique-flash-attention-pipeline`、`pattern-flash-attention-tiling-key-dispatch`。

## 1. 输入输出

### 输入（按 `prompt_flash_attention_def.cpp` L26-75；kernel 全量入参见 `prompt_flash_attention.cpp` L25-37）

| 输入 | 必选/可选 | 含义 | dtype |
|---|---|---|---|
| query / key / value | 必选 | Q/K/V | FP16/BF16/INT8（A2/950）；FP16（310P）；FP8/FP4（V3，部分 arch） |
| pse_shift (pse) | 可选 | 位置编码 | FP16/BF16/FLOAT |
| atten_mask | 可选 | 注意力掩码，true=遮蔽 | BOOL/UINT8/INT8/FLOAT16 |
| actual_seq_lengths / actual_seq_lengths_kv | 可选 | 变长真实序列长度 | INT64 |
| deq_scale1 / quant_scale1 / deq_scale2 / quant_scale2 / quant_offset2 | 可选 | 量化/反量化参数 | UINT64/FLOAT |
| blocktable | 可选 | PagedAttention 块表 | INT32 |
| key_antiquant_scale/offset, value_antiquant_scale/offset | 可选 | KV 反量化 | FP16/BF16/FLOAT |
| keySharedPrefix / valueSharedPrefix / actualSharedPrefixLen | 可选 | 共享前缀 KV | 同 KV |
| queryRope / keyRope | 可选 | 解耦 RoPE（MLA nope+rope） | 同 query |
| dequantScaleQuery / learnableSink | 可选 | Q 反量化 / 可学习 sink | FLOAT |

### 输出

| 输出 | 含义 | dtype |
|---|---|---|
| attention_out | 注意力输出 | FP16/BF16/INT8 |
| softmaxLse | log-sum-exp，供增量解码复用（kernel 入口 L36） | FLOAT |

注：训练 FA 输出 softmax_max + softmax_sum 供反向；PFA 输出 softmaxLse 供推理下游。精确 lse 公式以 `vf_softmaxflashv3.h` 为准（`needs_verification`）。

### 维度约定（README L94-166）
- B≤65536（D 非 16/32 对齐时仅 128）；N≤256；S≤20971520（20M）；D≤512。
- layout：BSH（默认）/SBH/BNSD/BSND/TND/NTD/PA_BSND。BNSD_BSND 时 query 为 BNSD、输出为 BSND。
- TND 约束（A2）：T≤65536，N∈{8,16,32,64,128}，Q_D=K_D=192，V_D=128/192，BF16，sparse=0(无mask)或3(有mask)。
- 长序列（B·S·N·D 乘积大）可能 aicore timeout，建议 S 切分（README L102）。

### 属性（def.cpp L80-103）
- `num_heads`（必需）、`num_key_value_heads`（默认0，GQA/MLA）。
- `scale_value`（默认1.0）、`pre_tokens`（默认214748647）、`next_tokens`（默认0）。
- `input_layout`（默认BSH）、`sparse_mode`（默认0）、`inner_precise`（默认1）。

## 2. 核心计算公式

```
attention_out = Softmax(scale * Mask(Q·K^T, atten_mask) [+pse]) · V
```
- pse add/mul 顺序由 pse_type 决定（PseMode：OUTER_MUL_ADD/OUTER_ADD_MUL/INNER_MUL_ADD/...，enum L189-194）。
- atten_mask 中 true 位置 select 成负极小值，softmax 后为 0 实现遮蔽（设计介绍 L11）。
- scale = scale_value（默认 1/√d 由调用方传入）。

## 3. 计算流程（在线 FlashSoftmax）

| 阶段 | 入口（arch35 service/） | 单元 | 操作 |
|---|---|---|---|
| Cube1 (Bmm1) | `prompt_flash_attention_mm1.h` | AIC | Q·K^T -> attention_score（FP32 累加） |
| Vector1 (Vec1) | `prompt_flash_attention_vector1.h` | AIV | pse -> ×scale -> atten_mask select -> FlashSoftmax(在线 max/sum) |
| Cube2 (Bmm2) | `prompt_flash_attention_mm2.h` | AIC | P·V matmul |
| Vector2 (Vec2) | `prompt_flash_attention_vector2.h` | AIV | rescale(÷softmax_sum) -> 可选 quant -> 写回 |

在线 FlashSoftmax 对 Skv 轴切分刷新（设计介绍 L13-22）：
- `exp[i] = e^(max[i-1] - max[i])`
- `PV[i] = PV[i-1]·exp[i] + PV_new[i]`（PV 累加器按 exp 缩放后累加新分块）
- `attention_out = PV_final / softmax_sum`
- 输出 `softmaxLse = log(softmax_sum) + max`（推理，`needs_verification` 精确公式）。

## 4. tiling 与分核（概要，详见 artifact `tiling_dispatch_tables.md`）

- 逻辑轴 B/N2/G/S1/S2，先核内再核间（设计介绍 L49）。
- Cube 基本块 128×128，Vector 8×1024，nRatio=8（1:16 CV 配比，设计介绍 L59）。
- S1 factor 默认 128、S2 factor 默认 1024（tiling.cpp L103-106）；值 source-reported，非最优声明。
- splitCoreMode：`actualSeqLength≥3072 && b·n·sOuter≥coreNum` -> SPLIT_ONEN_CUBE，否则 SPLIT_NBS_CUBE（tiling.cpp L926-932）；负载均衡 `singleCoreSize = ceil(totalSize/coreNum)`（L713）。
- MLA 触发：`D != Dv` 或 `D > 256`（tiling.cpp L2848），D=576(nope512+rope64)。
- 空 KV：tilingkey=20（tiling.cpp L73）。
- arch32 用十进制 tilingkey（~130 宏）；arch35 用结构化枚举模板分发（27 个 ConfigValue）。

## 5. 变体对照

| 变体 | 算子/版本 | 场景与差异 |
|---|---|---|
| 训练前向 | flash_attention_score | 训练，输出 softmax_max/sum 供反向 |
| **推理 prefill** | **prompt_flash_attention (V1)** | 全序列推理，输出 softmaxLse |
| V2 | aclnnPromptFlashAttentionV2 | 新增 MLA / PagedAttention / 量化扩展 |
| V3 | aclnnPromptFlashAttentionV3 | 新增 FP8/FP4 / learnableSink / 解耦 RoPE / sharedPrefix |
| 增量解码 | incre_flash_attention | KV cache 增量（PFA 的 prefill 结果供其复用 lse） |

所有变体共享 QK->FlashSoftmax->PV 骨架；PFA 相对训练 FA 的差异：推理语义、softmaxLse 输出、KV-cache 写回（`vf_flashupdate.h`，写回时机 `needs_verification`）、PagedAttention、量化（deq_scale/antiquant/MSD）、MLA、共享前缀、解耦 RoPE。

## 6. 架构与产品支持（README L5-12；def.cpp L114-127）

| 产品 | 架构 | 支持 | 说明 |
|---|---|---|---|
| Atlas A2 训练/推理 | Ascend910B (arch32) | √ | FP16/BF16/INT8；多模板 |
| Atlas A3 训练/推理 | Ascend910_93 (arch32) | √ | 复用 arch32 kernel |
| Ascend 950PR/950DT | Ascend950 (arch35) | √ | 服务化结构，模板参数化 |
| Atlas 推理系列加速卡 | Ascend310P | √ | 仅 FP16（`needs_verification`：精确芯片型号） |

FP8/FP4 仅 V3 且限定 arch；不支持 Atlas 200I/500 A2、Atlas 训练系列（910A）。

## 7. Agent actions

1. 确认目标架构（arch32 A2/A3 vs arch35 950）与 dtype/layout，选择对应模板分发路径。
2. 按 shape 评估 MLA 分支（D≠Dv 或 D>256）；MLA 下 Q/K 沿 D 拼接 nope+rope（D=576）。
3. 评估 S2 是否需 FlashSoftmax 在线刷新（S2 大于单块时）。
4. PagedAttention 场景：通过 blocktable 查物理 KV 偏移做间接 load。
5. 量化场景：按 QuantMode（per-channel/per-token/MSD）做 cast + scale 融合。
6. 参考 `technique-flash-attention-pipeline` 安排 CV 流水与 double buffer；参考 `technique-flash-attention-tiling` 的核内/核间切分原则。
7. Triton 实现参考 `artifacts/prompt-flash-attention/prompt-flash-attention-triton-skeleton.py`（pseudocode）。

## 8. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、流水 bound、UB/L1 占用、FP8/FP4 精度边界、softmaxLse 精确公式、KV-cache flashupdate 写回时机均未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 `ops-transformer-prompt-flash-attention`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Fused Attention / Flash Attention 高性能算子](../reference/op-class-attention.md) — 主题相关：共享 attention, flash_attention, softmax
- 实践案例: [Softmax / LogSoftmax 算子优化](../reference/op-class-softmax.md) — 主题相关：共享 online_softmax, softmax
- 实践案例: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 softmax
- 实践案例: [Online Softmax / Flash Attention 单遍稳定计算](../runbooks/algorithm-numerics/online-softmax-flash-attention.md) — 主题相关：共享 flash_attention, online_softmax
- 实践案例: [pad slot 单次检查提前退出，跳过无效填充序列](../runbooks/compute-pipeline/pad-slot-early-exit.md) — 主题相关：共享 kv_cache
- 实践案例: [persistent kernel 模式（grid-stride 摊还启动开销）](../runbooks/compute-pipeline/persistent-kernel-pattern.md) — 同属优化知识，主题相关
- 实践案例: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../runbooks/meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../runbooks/meta-skills/quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Flash Attention Causal Skeleton](../reference/skeleton-flash_attention_causal.md) — 同属优化知识，主题相关
- 相关主题: [FlashAttention 变体的内存访问优化](../runbooks/algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 attention, flash_attention, online_softmax, paged_attention
- 相关主题: [大序列 Attention 的分块计算与 Online Softmax 数值稳定](../runbooks/algorithm-numerics/traj-attention-memory_numerical.md) — 主题相关：共享 attention, flash_attention, online_softmax
- 相关主题: [Causal Conv1D Prefill/Update 跨 Kernel 优化迁移](../runbooks/compute-pipeline/traj-conv-cross_kernel.md) — 主题相关：共享 prefill
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](../runbooks/compute-pipeline/traj-conv-scheduling.md) — 同属优化知识，主题相关
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](../runbooks/compute-pipeline/traj-rope-scheduling.md) — 主题相关：共享 kv_cache
- 相关主题: [Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐](../runbooks/meta-skills/traj-attention-workflow_opt.md) — 主题相关：共享 attention, softmax
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, flash_attention, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, cube_vector_pipe, flash_attention, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, online_softmax, softmax, tail_mask, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — FlashAttentionScore 与 prompt 场景变体同属 attention 家族
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, softmax, tail_mask, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, flash_attention, fp32_accumulation, kv_cache, l0c_buffer, l1_buffer, online_softmax, paged_attention, tail_mask, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — FlashAttention 的 prompt/incremental 场景变体
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, flash_attention, fp32_accumulation, kv_cache, l0c_buffer, mla, quant, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, kv_cache, l0c_buffer, l0c_to_ub, l1_buffer, mla, quant, tail_mask, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, mla, prefill, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, kv_cache, l0c_buffer, l0c_to_ub, l1_buffer, mla, quant, tail_mask, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, softmax, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, softmax, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, quant, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, quant, unified_buffer
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, quant, unified_buffer
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, paged_attention, tail_mask, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, kv_cache, tail_mask, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, online_softmax, softmax, tail_mask, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, fp32_accumulation, online_softmax, softmax, unified_buffer
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, flash_attention, fp32_accumulation, kv_cache, l0c_buffer, l0c_to_ub, l1_buffer, mla, online_softmax, softmax, tail_mask, unified_buffer

<!-- okf:related:end -->
