---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: NsaCompressGrad (NSA compress backward) on Ascend NPU
description: CANN ops-transformer 中的 NsaCompressGrad 算子是 NSA（Native Sparse Attention） 压缩算子的反向计算。仅支持 Atlas A2 训练系列（arch32，ascend910b/91093），仅 TND layout。核心计算为两个 matmul 反向（dw = dkcmp · K^T，dk = W^T · dkcmp），纯 Vector 计算，无 Cube。
tags: [nsa_compress_grad, nsa, attention, backward, compress, sparse_attention, elementwise, reduce, fp32_accumulation, unified_buffer, ai_core, ascend910b1, ascend910b2, ascend910_93]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_compress_grad/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_compress_grad/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/nsa-compress-grad.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_compress_grad/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/nsa_compress_grad/)。

# NsaCompressGrad (NSA compress backward)

CANN ops-transformer 中的 `NsaCompressGrad` 算子是 [NSA（Native Sparse Attention）](https://arxiv.org/abs/2502.11089) 压缩算子的**反向计算**。仅支持 Atlas A2 训练系列（arch32，ascend910b/910_93），仅 TND layout。核心计算为两个 matmul 反向（`dw = dk_cmp · K^T`，`dk = W^T · dk_cmp`），纯 Vector 计算，无 Cube。

## 1. 输入输出

### 输入（按 `nsa_compress_grad_def.cpp` L22-49）

| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| outputGrad | 必选 | 正向算子输出的反向梯度（dk_cmp），shape [T, N, D] | BF16/FP16 |
| input | 必选 | 待压缩张量（K），shape [T, N, D] | BF16/FP16 |
| weight | 必选 | 压缩权重（W），shape [compressBlockSize, N]，与 input broadcast | BF16/FP16 |
| actSeqLen | 可选 | 每个 batch 的 S 大小（cumsum 格式，actSeqLenType=0） | INT64 |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| inputGrad | input 的梯度（dk），shape 同 input | BF16/FP16 |
| weightGrad | weight 的梯度（dw），shape 同 weight | BF16/FP16 |

### 属性

| 属性 | 含义 | 说明 |
|------|------|------|
| compressBlockSize | 压缩滑窗大小 | 必需，16 对齐 |
| compressStride | 两次压缩滑窗间隔大小 | 必需，16 对齐，compressBlockSize > compressStride |
| actSeqLenType | 序列长度类型 | 必需，0=cumsum 结果（当前仅支持 0） |
| layoutOptional | 输入排布格式 | 可选，默认 TND |

### 维度约束

- compressBlockSize 和 compressStride 必须是 16 的整数倍，且 compressBlockSize > compressStride（`tiling.cpp` L220-233）。
- headDim 必须是 16 的整数倍（`tiling.cpp` L231-233）。

## 2. 核心计算公式

```
dw = dk_cmp · K^T    # weight 梯度：沿 headDim 的 matmul reduce
dk = W^T · dk_cmp    # input 梯度：weight 沿 headDim 广播后逐元素乘
```

CANN kernel 实现：
- `ComputeWtGrad`（`nsa_compress_grad.h` L330-349）：`Mul(inputFp32Ub, cmpGrad)` → `BlockReduceSum`（headDim 64 分段）→ `WholeReduceSum`（合段）→ `AtomicAdd` 到 workspace。
- `ComputeInputGrad`（`nsa_compress_grad.h` L352-370）：`Cast(weight→FP32)` → `BroadCast（沿 headDim）` → `Mul(weightBrdc, cmpGrad)` → `CastByCondition（→输出 dtype）`。

## 3. 计算流程

### 3.1 分核（按 aivNum 沿 block 维外切）

```
totalBlockNum = outputGrad.shape[0]（压缩后的 block 数）
nBlockPerCore = totalBlockNum / aivNum
剩余块分配给前 nRemainder 个核
```

### 3.2 单核主循环

```
for each block assigned to this core:
    CopyInCmpGrad(block)              # GM→UB：当前 block 的 dk_cmp
    for each head in block:
        for each row in blockSize:
            CopyInInput(row)          # GM→UB：对应 input KV
            CopyInWt(row)             # GM→UB：对应 weight
            ComputeWtGrad()           # Mul + reduce → dw
            CopyOutWtGrad()           # AtomicAdd → workspace
            ComputeInputGrad()        # Cast + BroadCast + Mul → dk
            CopyOutInputGrad()        # AtomicAdd（overlap）或 store
```

### 3.3 核间规约

`MoveResult()`（`nsa_compress_grad.h` L419-437）：`SyncAll()` 后按 `deterministicFlag_` 分支：
- deterministic：`DeterministicComputeSumWtGrad`（树形规约）。
- 非 deterministic：`coreId_==0` 的核收集 AtomicAdd 结果。
- `isOverLap_`：`MoveInputGrad`（ping-pong buffer 搬运 workspace → output）。

## 4. 架构与产品支持

| 产品 | 架构 | config（def.cpp） | kernel 入口 | tiling |
|------|------|-------------------|-------------|--------|
| Atlas A2 训练 | arch32（910B） | `ascend910b` L72 | `nsa_compress_grad.cpp` | `tiling.cpp` |
| Atlas A3 训练 | ascend910_93 | `ascend910_93` L73 | 同上 | 同上 |

> 不支持 Ascend 950、310P、推理加速卡、A2 推理。TilingKey 恒为 0（单模板）。

## 5. 工作机制

### 5.1 compress 滑窗

正向 NSA compress 用滑动窗口对 input 压缩：每个窗口大小 `compressBlockSize`，步长 `compressStride`，输出 `blockNum` 个压缩后的 token。反向计算 `dk` 和 `dw`。

### 5.2 AtomicAdd 规约

多核可能写 `weightGrad` 同一位置（不同 block 的 weight 梯度累加），CANN 用 `SetAtomicAdd<float>` + `DataCopyPad` 实现原子加。Triton 可迁移为 `tl.atomic_add`。

### 5.3 overlap 处理

`compressBlockSize > compressStride` 时，相邻窗口有 overlap，input 的梯度需要累加。CANN 用 `isOverLap_` 标志控制 AtomicAdd 写 workspace 后再搬运。

### 5.4 变长 TND

`actSeqLenType=0` 表示 cumsum 格式，`GetSeqLenTarBch`（`nsa_compress_grad_base.h`）解析每个 batch 的序列长度。

## 6. Agent actions

1. 确认目标架构为 A2 训练（ascend910b/910_93），仅 TND layout。
2. 确认 `compressBlockSize` 与 `compressStride` 16 对齐且 `compressBlockSize > compressStride`。
3. 分核：`grid = (totalBlockNum,)` 沿 block 维外切。
4. 核心计算：`tl.dot(dk_cmp, K.T)`（weight 梯度）+ `tl.dot(W.T, dk_cmp)`（input 梯度）。
5. 多核写 `weightGrad` 同位置 → `tl.atomic_add`；overlap 场景 `inputGrad` 也需 atomic_add。
6. 变长支持：`actSeqLen` cumsum → per-batch offset + mask。
7. Triton 实现参考 `artifacts/nsa-compress-grad/nsa-compress-grad-triton-skeleton.py`（pseudocode）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、UB 字节预算、actSeqLenType=0 精确语义、AtomicAdd 多核精度边界、确定性规约算法差异需结合源码/硬件进一步验证。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 `ops-transformer-nsa-compress-grad`。

<!-- okf:related:start -->

# 相关

- 实践案例: [跨 tile 归约必须显式两阶段 reduce-then-apply](../runbooks/algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../runbooks/algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 attention, sparse_attention
- 相关主题: [Element-wise 算子避免冗余 Reshape](../runbooks/compute-pipeline/traj-general_perf-redundancy_elim.md) — 主题相关：共享 elementwise
- 相关主题: [Attention 硬件适配优化聚合](../runbooks/portability/traj-attention-hardware_adaptation.md) — 主题相关：共享 attention, sparse_attention
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, backward, fp32_accumulation, sparse_attention, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, sparse_attention, unified_buffer
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, elementwise, fp32_accumulation
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, backward, fp32_accumulation, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, attention, backward, fp32_accumulation, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, attention, fp32_accumulation, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, attention, fp32_accumulation, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, attention, fp32_accumulation, sparse_attention, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, sparse_attention, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, attention, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, elementwise, fp32_accumulation, reduce, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, elementwise, fp32_accumulation, reduce, unified_buffer
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, reduce, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, backward, fp32_accumulation, reduce, unified_buffer
- 相关主题: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](moe-token-permute-with-ep-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, reduce, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, backward, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](moe-token-unpermute-with-ep.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, reduce, unified_buffer
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, nsa, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 同一 NSACompressAttention 算子的前向与反向实现
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, elementwise, fp32_accumulation, nsa, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, fp32_accumulation, nsa, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, sparse_attention, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, elementwise, fp32_accumulation, unified_buffer
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, sparse_attention, unified_buffer

<!-- okf:related:end -->
