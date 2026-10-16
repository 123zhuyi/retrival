---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU
description: CANN ops-transformer 的 BlockSparseAttentionGrad 是 kernel-block-sparse-attention（前向，代号 RFA）的反向梯度算子，计算 dQ/dK/dV。共享前向的类 CUTLASS attninfra 模板库与 QK->softmax->PV 骨架（technique-flash-attention-pipeline），但有三处关键差异：(1) P 在反向重算（用前向保存的 softmaxLse，非在线...
tags: [block_sparse_attention, sparse_attention, backward, gradient, softmax_grad, atomic_add, recompute_p, gqa, sparse_attention_tiling, sparse_kv_gather, fp32_accumulation, tail_mask, unified_buffer, l1_buffer, l0a_buffer, l0b_buffer, l0c_buffer, cube_vector_1_to_2, mte2, mte3, buffer_id_sync, ascend910b1, ascend910_93, attention, block_sparse_attention_grad, out_of_memory]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/block_sparse_attention_grad/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/block_sparse_attention_grad/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/block-sparse-attention-grad.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/block_sparse_attention_grad/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/block_sparse_attention_grad/)。

# BlockSparseAttentionGrad（块稀疏注意力反向）

CANN ops-transformer 的 `BlockSparseAttentionGrad` 是 `kernel-block-sparse-attention`（前向，代号 RFA）的**反向梯度算子**，计算 dQ/dK/dV。共享前向的类 CUTLASS `attn_infra` 模板库与 QK->softmax->PV 骨架（`technique-flash-attention-pipeline`），但有三处关键差异：(1) **P 在反向重算**（用前向保存的 `softmaxLse`，非在线 softmax）；(2) **块稀疏 mask 改用门控稠密扫描**，不复用前向的 CSR 压缩 gather（`technique-block-sparse-mask-gather` 是前向策略，反向不同）；(3) **dQ/dK/dV 用 AtomicAdd 跨 KV 块累加**。仅训练场景。来源见 `ops-transformer-block-sparse-attention-grad`。

## 1. 输入输出

### 输入（def.cpp L19-62；docs L91-320）

| 输入 | 必选/可选 | 含义 | dtype | shape |
|---|---|---|---|---|
| dout | 必选 | 输出梯度 dY | FP16/BF16 | 同 query |
| query | 必选 | Q | FP16/BF16 | TND `[T,N1,D]` / BNSD `[B,N1,S,D]`；D=128 |
| key | 必选 | K | FP16/BF16 | TND `[T,N2,D]` / BNSD `[B,N2,S,D]` |
| value | 必选 | V | FP16/BF16 | 同 key |
| attentionOut | 必选 | 前向输出（softmaxGrad 用） | FP16/BF16 | 同 query |
| softmaxLse | 必选 | 前向 log-sum-exp（重算 P 用） | FLOAT | TND `[T,N1,1]` / BNSD `[B,N1,S,1]` |
| blockSparseMask | 必选 | 块稀疏 pattern | BOOL/UINT8 [needs_verification] | `[B,N1,ceilDiv(Q,X),ceilDiv(KV,Y)]` |
| attenMask | 必传 nullptr | - | - | 不支持 |
| blockShape | 可选 | `[X,Y]`，Y%128==0，默认 `[128,128]` | INT64 | - |
| actualSeqLengths / actualSeqLengthsKv | TND 必选 | 每 batch 实际长 | INT64 | - |

### 输出（def.cpp L63-74；infershape.cpp L82-84）

| 输出 | 含义 | dtype | shape |
|---|---|---|---|
| dq | dQ 梯度 | FP16/BF16 | 同 query |
| dk | dK 梯度 | FP16/BF16 | 同 key |
| dv | dV 梯度 | FP16/BF16 | 同 value |

### 属性（def.cpp L76-82）
- `qInputLayout`/`kvInputLayout`（TND/BNSD，须一致）、`numKeyValueHeads`、`maskType`（仅 0）、`scaleValue`、`preTokens`/`nextTokens`（仅 INT_MAX）。
- **无** `innerPrecise`、**无** `softmaxLseFlag`（前向专用；反向恒 FP32 softmax、恒消费 Lse）。

### 约束（README L361-367）
- q/k/v/dout dtype 一致（FP16/BF16），layout 一致，**headdim=128**。
- GQA：N1≥N2 && N1%N2==0（README L367 称当前版本仅 BNSD+MHA(N1==N2) [needs_verification]，TND/GQA 路径代码存在但文档受限）。
- 无 PagedAttention/滑窗/causal（maskType 仅 0）。

## 2. 核心反向公式

```
softmax_grad = softmaxGrad(dOut, attentionOut)        # = Σ(dY·O) per Q row
dP = dOut · V^T
dS = P · (dP − softmax_grad)
dV = P^T · dOut
dQ = (dS · K) · scale
dK = (dS^T · Q) · scale
```
- **P 重算**：`P = exp(S·scale − Lse)`，S=Q·K^T，用前向保存的 `softmaxLse`（`block_epilogue_simply_softmax.hpp` L335-393）。非在线 softmax。
- `attentionOut`（前向 Out=P·V）用于 softmax-grad 项 `D=Σ(dY·O)`（`block_epilogue_softmaxgrad.hpp` L208-297）。
- rescale：dQ/dK 写回乘 scale，**dV 不乘**（`block_epilogue_post.hpp` L371-374）。
- dQ/dK/dV 跨 KV 块 FP32 累加（AtomicAdd）。

## 3. 计算流程（4 阶段 cube↔vec + mask 门控扫描 + 重算 P）

骨架同 `technique-flash-attention-pipeline` 的反向；mask 处理与前向不同。

**块稀疏 mask 处理（与前向不同）**：反向**不复用**前向 `Mask2IdxAndCount` CSR 压缩（无 selectIdx/selectNumIdx）。cube 与 vec 核**均**直接读原始 bool mask，遍历所有 `kvBlockNum` 块，`if(mask)` 为真才计算、否则跳过（`kernel.h` L386-389,604-607）--**mask 门控稠密扫描**（branch-on-mask）。前向用 CSR 压缩 + gather（`technique-block-sparse-mask-gather`）；反向选择更简单的逐块分支，牺牲冗余 mask 扫描换调度简单（dQ/dK/dV 跨块累加使 CSR 重派生收益降低）[needs_verification]。

**4 阶段流水**（1-deep `preTaskInfo` delay，3 flag `CUBE2VEC=7`/`VEC2CUBE=8`/`CUBE2POST=9`）：

| 阶段 | 单元 | 入口 | 操作 |
|---|---|---|---|
| Stage1 cube | AIC | kernel.h L405-422 | S=Q·K^T + dP=dOut·V^T（blockMmad1）-> set CUBE2VEC |
| Stage2 vec | AIV | kernel.h L597-675 | wait CUBE2VEC；D=Σ(dY·O)；重算 P=exp(S·scale−Lse)；dS=P·(dP−D) -> set VEC2CUBE |
| Stage3 cube | AIC | kernel.h L423-511 | wait VEC2CUBE；dQ=dS·K、dK=dS^T·Q、dV=P^T·dOut（blockMmad2/3，**AtomicAdd**）-> set CUBE2POST |
| Stage4 vec | AIV | kernel.h L515-531 | wait CUBE2POST + SyncAll；rescale(dQ/dK*=scale)+cast+写回（BlockPost） |

3 组 cube GEMM 复用同一 `SBSAG2` 策略（`block_mmad_bsag2.hpp`），Cube2/Cube3 `isAtomicAdd=true` 累加 dQ/dK/dV。dQ/dK/dV 晚一拍（preTaskInfo delay）以重叠 vec softmax。

## 4. tiling 与分核（概要，详见 artifact tiling_dispatch_tables.md）

- **4 个 tiling key**（dtype×layout：0/1/10/11，`tiling.cpp` L497-514），抛弃前向 12-key RFA 方案；无 innerPrecise/lse/maskType/pagedCache 轴。反向 softmax 恒 FP32。
- 核间仅在 **(Q-block × head)** 切分：`blockDim=min(aicNum, totalTaskNum)`，`totalTaskNum=Σ_b ceilDiv(qseqlen_b, blockShapeX)·numHeads`。**无 FlashDecoding、无 S2 切分**。
- 每核任务描述子数组 `beginBatch/beginHead/beginQSeqOffset/preQSeqLengths/preKVSeqLengths[64]`（与前向 sub-core 分区不同）。
- GQA：`groupSize=numHeads/kvHeads`，KV head 每 groupSize 个 Q head 推进。
- 基本块 `BASIC_BLOCK_SIZE=128`、`basicQBlockSize=basicKVBlockSize=128`；非对齐 ceilDiv + tail。
- workspace（FP32）：`[gS(重算S) | gDp(dP) | gDq | gDk | gDv | sftmg(softmax-grad)]`（L422-447），**无** selectIdx/selectNumIdx。
- softmax-grad tiling：`AscendC::SoftMaxGradTilingFunc`（L278-300）。

## 5. 变体对照

| 变体 | 算子 | mask 处理 | P 来源 | 累加 |
|---|---|---|---|---|
| 前向 | block_sparse_attention | CSR 压缩 + gather | 在线 softmax | PV 累加 |
| **反向** | **block_sparse_attention_grad** | **门控稠密扫描** | **重算（用 Lse）** | **dQ/dK/dV AtomicAdd** |
| 参考 | flash_attention_score_grad | atten_mask | 重算/保存 | atomic |

相对前向的核心差异：mask 门控稠密扫描（非 CSR）、重算 P（非在线 softmax）、4 阶段顺序流水（非交错在线）、3 cube GEMM（S&dP/dQ/dK&dV，非 QK/PV）、AtomicAdd 累加、4 tiling key（非 12）、训练专用。

## 6. 架构支持（docs L5-12；def.cpp L84-85）

| 产品 | 架构 | 支持 |
|---|---|---|
| Atlas A2 训练 | Ascend910B1 | √ |
| Atlas A3 训练 | Ascend910_93 | √ |

**仅训练**：A2 推理 ×、A3 推理 ×（前向 A2/A3 训练+推理均 √，反向训练专用）。不支持 950/310P/其余。仅 FP16/BF16。

## 7. Agent actions

1. 按 `qInputLayout`/`kvInputLayout`（TND/BNSD，须一致）解析地址；D=128。
2. **重算 P**：`P=exp(tl.dot(q,k)*scale - tl.load(lse))`，用前向保存 `softmaxLse`；`attentionOut` 用于 `D=Σ(dY·O)`。
3. **块稀疏 mask 门控扫描**：遍历 KV 块，`m=tl.load(mask); if m==0: continue`（反向不复用前向 CSR；亦可选 CSR gather，取舍见 `technique-block-sparse-mask-gather` [needs_verification]）。
4. 反向 GEMM：dP=dY·V^T、dS=P·(dP−D)、dQ+=dS·K（核内 FP32 累加）、dK+=dS^T·Q、dV+=P^T·dY（dK/dV 跨程序 `tl.atomic_add` 同 KV）。
5. rescale：dQ/dK 乘 scale，**dV 不乘**。
6. GQA：`groupSize=numHeads/kvHeads`，KV head 跨 Q head 共享。
7. 核间切分：program 维 = (batch, head, Q-block)；KV 不切核。`BLOCK_M=128`（source-reported）。
8. Triton 实现参考 `artifacts/block-sparse-attention-grad/block-sparse-attention-grad-triton-skeleton.py`（pseudocode）。CANN 专有机制（MTE/CrossCore flag/preTaskInfo delay/TQue/AtomicAdd/SoftmaxGradFront）不迁移。

## 8. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、UB/L1 占用、mask 门控 vs CSR 的稀疏度取舍、AtomicAdd 数值精度、mask dtype 不一致（UINT8/BOOL/bool）、headdim 强制、TND+GQA 支持范围、任务类型宏缺失、scaleValue=0.0 默认、SBSAG1/SBSAG3 未用等均未经硬件实测，见 artifact migration-table.md 的 `needs verification` 项。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 `ops-transformer-block-sparse-attention-grad`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Fused Attention / Flash Attention 高性能算子](../reference/op-class-attention.md) — 主题相关：共享 attention, gqa
- 实践案例: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../runbooks/algorithm-numerics/atomic-add-cross-tile-reduction.md) — 主题相关：共享 atomic_add
- 实践案例: [原子操作粒度--连续段 E>=64 用向量 atomic_add，禁止标量循环 atomic](../runbooks/algorithm-numerics/atomic-granularity.md) — 同属优化知识，主题相关
- 实践案例: [care_padding=False 消除 MTE2→Vector 同步屏障](../runbooks/compute-pipeline/care-padding-false-ilp.md) — 主题相关：共享 mte2
- 实践案例: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](../runbooks/compute-pipeline/grid-limit-65535.md) — 同属优化知识，主题相关
- 实践案例: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](../runbooks/compute-pipeline/load-balance-formula.md) — 同属优化知识，主题相关
- 实践案例: [swizzle2d 块重排提升 L2 复用](../runbooks/compute-pipeline/swizzle2d-block-reorder.md) — 同属优化知识，主题相关
- 实践案例: [Work-Stealing 动态负载均衡（不均匀任务分发）](../runbooks/compute-pipeline/work-stealing-dynamic-load-balance.md) — 主题相关：共享 atomic_add
- 实践案例: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../runbooks/memory-sram/block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 实践案例: [M>=256 时 BLOCK_M=256（总 block-iter 减半）](../runbooks/memory-sram/blockm-256-large-m.md) — 同属优化知识，主题相关
- 实践案例: [BLOCK 向上取 2 的幂并声明为 constexpr](../runbooks/memory-sram/pow2-block-constexpr.md) — 同属优化知识，主题相关
- 实践案例: [UB 占用决定 BLOCK_ROWS/tile 大小，必须按算子特征分别自适应](../runbooks/memory-sram/ub-aware-block-adaptive.md) — 同属优化知识，主题相关
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../runbooks/algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 attention, gqa, sparse_attention
- 相关主题: [FlashAttention 变体的内存访问优化](../runbooks/algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 attention, gqa
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../runbooks/compute-pipeline/traj-conv-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [Scatter Weight Gradient 向量化计算优化](../runbooks/compute-pipeline/traj-gather_scatter-compute.md) — 同属优化知识，主题相关
- 相关主题: [Triton Block Size 通用调优原则](../runbooks/compute-pipeline/traj-general_perf-block_size.md) — 同属优化知识，主题相关
- 相关主题: [MoE 稀疏 Weight Gradient 计算优化](../runbooks/compute-pipeline/traj-moe-compute.md) — 同属优化知识，主题相关
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](../runbooks/memory-sram/traj-embedding-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../runbooks/memory-sram/traj-general_perf-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [Attention 硬件适配优化聚合](../runbooks/portability/traj-attention-hardware_adaptation.md) — 主题相关：共享 attention, sparse_attention
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 同一 BlockSparseAttention 算子的前向与反向实现
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, backward, cube_vector_1_to_2, fp32_accumulation, gradient, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, backward, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, sparse_attention, sparse_attention_tiling, sparse_kv_gather, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, sparse_attention, sparse_kv_gather, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, gradient, unified_buffer
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, gradient, unified_buffer
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, backward, fp32_accumulation, sparse_attention, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, tail_mask, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, gqa, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, sparse_attention, sparse_attention_tiling, tail_mask, unified_buffer
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — 主题相关：共享 ascend910b1, tail_mask
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, sparse_attention, sparse_attention_tiling, sparse_kv_gather, tail_mask, unified_buffer

<!-- okf:related:end -->
