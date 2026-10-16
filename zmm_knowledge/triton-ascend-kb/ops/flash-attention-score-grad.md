---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: FlashAttentionScoreGrad training-backward on Ascend NPU
description: CANN ops-transformer 中的 FlashAttentionScoreGrad 算子（简称 FAG）在训练场景下计算 FlashAttentionScore 的反向梯度，支持 Atlas A2 训练系列（arch32 / ascend910b / ascend91093）与 Ascend 950PR/950DT（arch35）。本页归纳反向算子语义与计算流程；可复用的 tiling/流水/dispatch 知识见 technique-flash-atte...
tags: [flash_attention, attention, softmax, online_softmax, backward, gradient, matmul, cube_vector_pipe, recompute, fp32_accumulation, tail_mask, unified_buffer, l0c_to_ub, cube_vector_1_to_2, l1_buffer, l0c_buffer, ascend910b1, ascend910_93, ascend950pr, flash_attention_score_grad]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/flash_attention_score_grad/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/flash_attention_score_grad/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/flash-attention-score-grad.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/flash_attention_score_grad/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/flash_attention_score_grad/)。

# FlashAttentionScoreGrad (训练反向)

CANN ops-transformer 中的 `FlashAttentionScoreGrad` 算子（简称 FAG）在训练场景下计算 `FlashAttentionScore` 的反向梯度，支持 Atlas A2 训练系列（arch32 / ascend910b / ascend910_93）与 Ascend 950PR/950DT（arch35）。本页归纳反向算子语义与计算流程；可复用的 tiling/流水/dispatch 知识见 `technique-flash-attention-tiling`、`technique-flash-attention-pipeline`、`pattern-flash-attention-tiling-key-dispatch`，反向重算方法见 `technique-flash-attention-backward-recompute`。前向算子见 `kernel-flash-attention-score`。

## 1. 输入输出

### 输入（按 op_host/flash_attention_score_grad_def.cpp 顺序）

| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| query | 必选 | 查询 Q | FP16/BF16/FLOAT/HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN |
| key | 必选 | 键 K | 同 query |
| value | 必选 | 值 V | 同 query |
| dy | 必选 | 输出梯度 dY | FP16/BF16/FLOAT（FP8 路径下同 qkv） |
| pse_shift | 可选 | 位置编码 | FP16/BF16/FLOAT |
| drop_mask | 可选 | dropout 掩码，1=保留 0=丢弃 | UINT8 |
| padding_mask | 可选 | padding 掩码 | FP16/BF16/FLOAT |
| atten_mask | 可选 | 注意力掩码，1=不参与 0=参与 | BOOL/UINT8 |
| softmax_max | 可选 | 前向 softmax 行最大值，加速重算 | FLOAT |
| softmax_sum | 可选 | 前向 softmax 行求和，加速重算 | FLOAT |
| softmax_in | 可选 | 前向 softmax 后的 P | FP16/BF16/FLOAT |
| attention_in | 可选 | 前向输出 Y=PV，参与 FlashSoftmaxGrad | FP16/BF16/FLOAT |
| prefix / actual_seq_qlen / actual_seq_kvlen / q_start_idx / kv_start_idx | 可选 | 变长/偏移（TND、varlen） | INT64 |
| d_scale_q/k/v/dy/o | 可选 | FP8 全量化反量化参数 | FLOAT |
| query_rope / key_rope | 可选 | RoPE 编码张量 | 同 query |
| sink / ds_scale / p_scale | 可选 | attention sink / P 缩放（V3+） | FLOAT |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| dq / dk / dv | Q/K/V 的梯度 | FP16/BF16/FLOAT |
| dpse | pse 的梯度 | FP16/BF16/FLOAT |
| dq_rope / dk_rope | RoPE 梯度（V3+） | FP16/BF16/FLOAT |
| dsink | sink 梯度（V3+） | FLOAT |

### 维度约定（README L187-194）

以 BNSD/BSND 为例（BSH/SBH 下 H=N*D）：B: 1~2M（prefix 时最大 2K）；N: 1~256；S: 1~1M；D: A2/A3 训练 1~512，950PR 1~768。内核计算逻辑轴：B、N2（kv head）、G（query/kv head 比）、S1（query seq）、S2（kv seq）、D、Dv。

### 属性（def.cpp L222-260）

`scale_value`(1.0)、`keep_prob`(1.0)、`pre_tockens`(INT_MAX)、`next_tockens`(INT_MAX)、`head_num`(必需)、`input_layout`(必需 BSH/SBH/BNSD/BSND/TND)、`inner_precise`(0)、`sparse_mode`(0，0=NO_MASK..8=BAND_LEFT_UP_CASUAL)、`pse_type`(1)、`seed`(2)、`offset`(0)、`out_dtype`(0)、`softmax_in_layout`("")。

## 2. 核心反向公式（6 步，FAG算子设计介绍.md L9-14）

设 `S=Mask(QK^T/√d + pse, atten_mask)`，`P=Dropout(Softmax(S),keep_prob)`，`Y=PV`：

| 步 | 公式 | 说明 |
|----|------|------|
| 1 重算 p | `p = SimpleSoftmax(Mask(Matmul(query, key^T) + pse) * scale)` | 重算前向 softmax 结果 P，存 UB |
| 2 dp | `dp = Dropout(Matmul(dy, value^T))` | dY·V^T |
| 3 ds | `ds = p * Sub(dp, FlashSoftmaxGrad(dy, attention_in))` | FlashSoftmaxGrad 入参 dY 与前向输出 Y |
| 4 dq | `dq = Matmul(ds, key) * scale` | ds·K |
| 5 dk | `dk = Matmul(ds^T, query) * scale` | ds^T·Q |
| 6 dv | `dv = Matmul(Dropout(p)^T, dy)` | Dropout(P)^T·dY |

FlashSoftmaxGrad 实质：`ds = P ⊙ (dP − rowsum(dP ⊙ P))`，`dP = dY·V^T`（步 2）。反向**重算** P（softmax_max/sum 作为可选输入可加速重算，见 def.cpp L72-83）。共 4 次 matmul：mm1=Q·K^T（重算）、mm2=dy·V^T、mm3=ds·K / ds^T·Q、mm4=P^T·dy。

## 3. 计算流程（3 阶段 Pipeline，FAG算子设计介绍.md L20-23）

| 阶段 | 入口函数 | 单元 | 对应步 |
|------|----------|------|--------|
| Cube1 | ComputeMM1 | AIC | 步 1（重算 p：QK^T + softmax） |
| Vector1 | ComputeVec | AIV | 步 2-3（dp、ds：FlashSoftmaxGrad + elementwise） |
| Cube2 | ComputeMMDqkv | AIC | 步 4-6（dq/dk/dv matmul） |

V 侧 double buffer（ping-pong）；CV 间 preload（Cube 提前发射）。arch32 C:V=1:2 核数比，1:16 数据配比（nRatio=8）；arch35 C:V=1:2，L0C->UB / UB->L1 高速通路。详见 `technique-flash-attention-pipeline`。

## 4. Tiling 与多模板

- **arch32**：8 类模板，按 shape 路由（FAG算子设计介绍.md L264-319）：S1S2（通用泛化）、SameAB（_sab，Cube 为主核，确定性/大序列）、BN2（S1<1024&&S2<1024&&(B*N2*2>CoreNum‖G==1)）、NGS1S2_BN（G*S1*S2≤32KB&&G==1&&S2<1536）、B（N2*G*S1*S2≤64*128）、Basic/MLA、Basic_det（确定性）、Empty。CV 基本块 C 128×1024 / V 8×1024。
- **arch35**：4 类模板（BN2: S1≤128&&S2≤128&&G==1；BN2S2: TND&&B*N2*S2>cubeCoreNum&&G==1；BN2GS1S2 通用；确定性），仅 2 个 kernel 文件（非确定性/确定性），模板参数隔离。CV 基本块 C 128×128 / V 64×128。
- 切分顺序：先核内再核间；S2 不切多核（softmax 在线更新需连续 S2）。详见 `technique-flash-attention-tiling`。

## 5. 模板 dispatch（op_kernel/flash_attention_score_grad.cpp L488-901）

20 个 kernel 模板参数，按 `UB0/UB1/Block/IsSameAB/IsDeterministic` 分发：空 tensor(0,0,0) / SameAB(4,3,4,1 → key 0x1434) / S1S2(4,3,4,0) / BN2(4,3,1 → 0x134) / NGS1S2_BN(9,9,1 → 0x199) / B(9,9,0 → 0x099) / Basic_det(9,9,9,Det=1 → 0x1999) / Basic(9,9,9 → 0x999)。按 dtype（FP16/BF16/FLOAT）特化，FP32 仅 S1S2/BN2/BN/B。dispatch 维度（dtype/layout/has_mask/has_pse/has_dropout/has_rope/deterministic/sparse/NZ 输出/L1 复用/对齐特化）可映射为 Triton `tl.constexpr` 分支与 `BLOCK_*` 选择。详见 artifact `tiling-dispatch-tables.md` 与 `pattern-flash-attention-tiling-key-dispatch`。

## 6. 架构与产品支持

| 产品 | 架构代号 | 支持 | 说明 |
|------|----------|------|------|
| Atlas A2 训练 | arch32 (ascend910b) | √ | AIC&AIV 分离，多模板（8 类），aclnn V1-V3 |
| Atlas A3 训练 | ascend910_93 | √ | 复用 arch32 kernel |
| Ascend 950PR/950DT | arch35 (DAV_3510) | √ | AIC&AIV 融合，统一模板，aclnn V4 |
| Atlas A2/A3 推理 | - | × | 仅训练场景 |

FP8（FLOAT8_E5M2/E4M3FN/HIFLOAT8）仅 arch35(950) 支持，且不可配 pse/dropMask/attenMask/keepProb（README L179-181）。aclnn 版本：V1-V3 仅 A2 训练（950PR ×），V4 仅 950PR（A2 ×）；V2 加 pseType，V3 加 sink/dsink，V4 合并 V2+UnpaddingV2+varlen+自动 Dropout（docs/aclnnFlashAttentionScoreGrad*.md）。

## 7. Agent actions

1. 确认架构（arch32 vs arch35）与 dtype，选择对应模板分发路径与 aclnn 版本（V1-V3 A2 / V4 950）。
2. 反向需重算 P：按 `technique-flash-attention-backward-recompute` 安排 6 步计算与 4 次 matmul。
3. 按输入 layout 与 shape 推导核内/核间切分轴，参考 `technique-flash-attention-tiling`；S2 不切多核。
4. 检查 atten_mask/dropMask/pse/softmax_max/sum/attention_in 是否存在，选对应 tiling key bit / constexpr 分支。
5. 参考 `technique-flash-attention-pipeline` 安排 Cube/Vector 流水与 double buffer；arch35 利用 L0C->UB/UB->L1。
6. Triton 实现参考 `artifacts/flash-attention-score-grad/flash-attention-score-grad-triton-skeleton.py`（pseudocode）。
7. 空 tensor 走短路清零（FAG_EMPTY_TILING_KEY，100MB workspace）。

## 8. 复现性声明

本页结论基于源码静态分析（confidence: inferred），tiling 参数数值、流水 bound、FP8/BF16 精度边界、aclnn 运行时契约未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 `ops-transformer-flash-attention-score-grad`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Fused Attention / Flash Attention 高性能算子](../reference/op-class-attention.md) — 主题相关：共享 attention, flash_attention, softmax
- 实践案例: [Embedding / 索引拷贝算子优化](../reference/op-class-embedding.md) — 主题相关：共享 backward
- 实践案例: [MatMul / GEMM 高性能算子](../reference/op-class-matmul.md) — 主题相关：共享 matmul
- 实践案例: [归一化算子优化](../reference/op-class-normalization.md) — 主题相关：共享 softmax
- 实践案例: [RoPE（旋转位置编码）算子优化](../reference/op-class-rope.md) — 主题相关：共享 backward
- 实践案例: [Softmax / LogSoftmax 算子优化](../reference/op-class-softmax.md) — 主题相关：共享 online_softmax, softmax
- 实践案例: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 softmax
- 实践案例: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](../runbooks/algorithm-numerics/exp_stability_001.md) — 同属优化知识，主题相关
- 实践案例: [Online Softmax / Flash Attention 单遍稳定计算](../runbooks/algorithm-numerics/online-softmax-flash-attention.md) — 主题相关：共享 flash_attention, online_softmax
- 实践案例: [matmul core mapping 策略选择（列/行/对角）](../runbooks/compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 实践案例: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../runbooks/compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 实践案例: [流程优化: 标准 score-归一化-加权三步流程](../runbooks/compute-pipeline/exp_fused_attention_001.md) — 主题相关：共享 attention
- 实践案例: [Scheduling Optimization: hardware unit isolation](../runbooks/compute-pipeline/exp_fused_norm_rope_matmul_001.md) — 主题相关：共享 matmul
- 实践案例: [计算优化: fixpipe 随路量化 + K 维循环展开 + 预取](../runbooks/compute-pipeline/exp_matmul_002.md) — 主题相关：共享 matmul
- 实践案例: [Compute Optimization: vectorized accumulation strategy](../runbooks/compute-pipeline/exp_softmax_inner_002.md) — 主题相关：共享 softmax
- 实践案例: [Compute Optimization: online softmax algorithm](../runbooks/compute-pipeline/exp_softmax_non_inner_001.md) — 主题相关：共享 softmax
- 实践案例: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../runbooks/memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 实践案例: [Memory Access Optimization: online softmax](../runbooks/memory-sram/exp_attn_sinks_001.md) — 主题相关：共享 attention
- 实践案例: [Memory Access Optimization: online softmax algorithm](../runbooks/memory-sram/exp_flashattn_fwd_001.md) — 主题相关：共享 attention
- 实践案例: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](../runbooks/memory-sram/exp_fused_attention_002.md) — 主题相关：共享 attention
- 实践案例: [访存优化：大尺寸 matmul 的 256 分块与 L0C 复用](../runbooks/memory-sram/exp_matmul_001.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块](../runbooks/memory-sram/exp_matmul_003.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块](../runbooks/memory-sram/exp_matmul_004.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：双转置矩阵乘的 512B 对齐分块](../runbooks/memory-sram/exp_matmul_005.md) — 主题相关：共享 matmul
- 实践案例: [Memory Access Optimization: score-tile reuse with online softmax](../runbooks/memory-sram/exp_sdpa_flash_001.md) — 主题相关：共享 attention
- 实践案例: [Memory Access Optimization: two pass reduction strategy](../runbooks/memory-sram/exp_softmax_backward_001.md) — 主题相关：共享 softmax
- 实践案例: [Memory Access Optimization: two pass reduction strategy](../runbooks/memory-sram/exp_softmax_backward_non_inner_001.md) — 主题相关：共享 softmax
- 实践案例: [Cube-Vector 场景行宽必须 512B 对齐](../runbooks/memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 实践案例: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../runbooks/meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [tl.dot vs torch.matmul 精度差异诊断](../runbooks/meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Flash Attention Causal Skeleton](../reference/skeleton-flash_attention_causal.md) — 同属优化知识，主题相关
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, backward, cube_vector_1_to_2, fp32_accumulation, gradient, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, flash_attention, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](chunk-gated-delta-rule-fwd-h.md) — 主题相关：共享 matmul
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 ascend910b1, ascend950pr, fp32_accumulation, tail_mask
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 同一 FlashAttentionScore 算子的前向与反向实现
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, backward, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, matmul, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, softmax, tail_mask, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, flash_attention, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, cube_vector_pipe, flash_attention, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, flash_attention, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, matmul, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, backward, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, softmax, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, softmax, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, backward, fp32_accumulation, gradient, unified_buffer
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, backward, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, gradient, unified_buffer
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, backward, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, tail_mask, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, cube_vector_pipe, flash_attention, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, online_softmax, softmax, tail_mask, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, online_softmax, softmax, tail_mask, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, fp32_accumulation, online_softmax, softmax, unified_buffer
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, flash_attention, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, online_softmax, softmax, tail_mask, unified_buffer

<!-- okf:related:end -->
