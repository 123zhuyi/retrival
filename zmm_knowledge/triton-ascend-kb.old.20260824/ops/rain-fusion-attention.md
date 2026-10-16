---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU
description: CANN ops-transformer 的 RainFusionAttention 是基于 CATLASS 模板库的高性能块级稀疏推理注意力算子（从 catlass 32sparseattentioninfer 迁移）。与 kernel-sparse-flash-attention 一样使用稀疏 QK/PV，但稀疏模式由 selectIdx（显式 Q×KV 块选择矩阵）驱动，比 SFA 的 sparseindices 更灵活（README L138-163）。注意：R...
tags: [rain_fusion, sparse_attention, attention, softmax, online_softmax, matmul, catlass, block_sparse, cube_vector_pipe, fp32_accumulation, tail_mask, sparse_attention_tiling, unified_buffer, l0c_to_ub, cube_vector_1_to_2, l1_buffer, l0c_buffer, ascend910b1, ascend910_93, rain_fusion_attention]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/rain_fusion_attention/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/rain_fusion_attention/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/rain-fusion-attention.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/rain_fusion_attention/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/rain_fusion_attention/)。

# RainFusionAttention（CATLASS 块级稀疏推理注意力）

CANN ops-transformer 的 `RainFusionAttention` 是基于 **CATLASS 模板库**的高性能块级稀疏推理注意力算子（从 catlass `32_sparse_attention_infer` 迁移）。与 `kernel-sparse-flash-attention` 一样使用稀疏 QK/PV，但稀疏模式由 `selectIdx`（显式 Q×KV 块选择矩阵）驱动，比 SFA 的 `sparse_indices` 更灵活（README L138-163）。**注意**：README 说 BNSD 不支持，但源码 tilingkey.h 和 kernel 入口定义了 BNSD 分支——以源码为准（`needs_verification`）。

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

<!-- okf:related:start -->

# 相关

- 实践案例: [融合算子综合页](../reference/op-class-fused_ops.md) — 主题相关：共享 attention
- 实践案例: [MatMul / GEMM 高性能算子](../reference/op-class-matmul.md) — 主题相关：共享 matmul
- 实践案例: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 softmax
- 实践案例: [Online Softmax / Flash Attention 单遍稳定计算](../runbooks/algorithm-numerics/online-softmax-flash-attention.md) — 主题相关：共享 online_softmax
- 实践案例: [dl.async_task 实现 Cube/Vector 异构并行](../runbooks/compute-pipeline/async-task-cube-vector.md) — 同属优化知识，主题相关
- 实践案例: [matmul core mapping 策略选择（列/行/对角）](../runbooks/compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 实践案例: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../runbooks/compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 实践案例: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](../runbooks/compute-pipeline/grid-limit-65535.md) — 同属优化知识，主题相关
- 实践案例: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](../runbooks/compute-pipeline/load-balance-formula.md) — 同属优化知识，主题相关
- 实践案例: [算子融合模式与约束决策](../runbooks/compute-pipeline/operator-fusion-patterns.md) — 同属优化知识，主题相关
- 实践案例: [persistent kernel 模式（grid-stride 摊还启动开销）](../runbooks/compute-pipeline/persistent-kernel-pattern.md) — 同属优化知识，主题相关
- 实践案例: [swizzle2d 块重排提升 L2 复用](../runbooks/compute-pipeline/swizzle2d-block-reorder.md) — 同属优化知识，主题相关
- 实践案例: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../runbooks/memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 实践案例: [M>=256 时 BLOCK_M=256（总 block-iter 减半）](../runbooks/memory-sram/blockm-256-large-m.md) — 同属优化知识，主题相关
- 实践案例: [BLOCK 向上取 2 的幂并声明为 constexpr](../runbooks/memory-sram/pow2-block-constexpr.md) — 同属优化知识，主题相关
- 实践案例: [Cube-Vector 场景行宽必须 512B 对齐](../runbooks/memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 实践案例: [UB 占用决定 BLOCK_ROWS/tile 大小，必须按算子特征分别自适应](../runbooks/memory-sram/ub-aware-block-adaptive.md) — 同属优化知识，主题相关
- 实践案例: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../runbooks/meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 attention
- 实践案例: [tl.dot vs torch.matmul 精度差异诊断](../runbooks/meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../runbooks/meta-skills/quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../runbooks/algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 attention, online_softmax, sparse_attention
- 相关主题: [大序列 Attention 的分块计算与 Online Softmax 数值稳定](../runbooks/algorithm-numerics/traj-attention-memory_numerical.md) — 主题相关：共享 attention, online_softmax
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../runbooks/algorithm-numerics/traj-normalization-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](../runbooks/compute-pipeline/instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 matmul
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../runbooks/compute-pipeline/traj-conv-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [Triton Block Size 通用调优原则](../runbooks/compute-pipeline/traj-general_perf-block_size.md) — 同属优化知识，主题相关
- 相关主题: [Matmul Scheduling 优化聚合](../runbooks/compute-pipeline/traj-matmul-scheduling.md) — 主题相关：共享 matmul
- 相关主题: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](../runbooks/compute-pipeline/traj-matmul-split.md) — 主题相关：共享 matmul
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](../runbooks/memory-sram/traj-embedding-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../runbooks/memory-sram/traj-general_perf-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐](../runbooks/meta-skills/traj-attention-workflow_opt.md) — 主题相关：共享 attention, softmax
- 相关主题: [Attention 硬件适配优化聚合](../runbooks/portability/traj-attention-hardware_adaptation.md) — 主题相关：共享 attention, sparse_attention
- 相关主题: [Matmul 硬件适配优化聚合](../runbooks/portability/traj-matmul-hardware_adaptation.md) — 主题相关：共享 matmul
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, sparse_attention, sparse_attention_tiling, tail_mask, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, sparse_attention, sparse_attention_tiling, tail_mask, unified_buffer
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, online_softmax, softmax, tail_mask, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, online_softmax, softmax, tail_mask, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, matmul, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, softmax, tail_mask, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, sparse_attention, sparse_attention_tiling, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, sparse_attention, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, matmul, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, softmax, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, softmax, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, sparse_attention, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, tail_mask, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, online_softmax, tail_mask, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, online_softmax, softmax, tail_mask, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, online_softmax, softmax, unified_buffer
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, online_softmax, softmax, sparse_attention, sparse_attention_tiling, tail_mask, unified_buffer

<!-- okf:related:end -->
