---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU
description: CANN ops-transformer 的 MlaProlog 是推理场景 MLA 前处理融合算子 V1（与 kernel-mla-preprocess V2+ 不同）。四路计算：x→×W^DQ(RmsNorm)→[×W^UQ·W^UK→q^N]+[×W^QR→RoPE→q^R]+[×W^DKV→RmsNorm→cache→k^C]+[×W^KR→RoPE→cache→k^R]。它是 kernel-sparse-flash-attention/kernel-promp...
tags: [mla, attention, rmsnorm, matmul, rope, kv_cache, fusion, cube_vector_pipe, quant, fp8, fp32_accumulation, tail_mask, unified_buffer, l0c_to_ub, cube_vector_1_to_2, l1_buffer, l0c_buffer, ascend910b1, ascend910_93, ascend950pr, mla_prolog]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/mla_prolog/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/mla_prolog/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/mla-prolog.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/mla_prolog/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/mla_prolog/)。

# MlaProlog（MLA 前处理 Prolog V1）

CANN ops-transformer 的 `MlaProlog` 是**推理场景** MLA 前处理融合算子 V1（与 `kernel-mla-preprocess` V2+ 不同）。四路计算：x→×W^DQ(RmsNorm)→[×W^UQ·W^UK→q^N]+[×W^QR→RoPE→q^R]+[×W^DKV→RmsNorm→cache→k^C]+[×W^KR→RoPE→cache→k^R]。它是 `kernel-sparse-flash-attention`/`kernel-prompt-flash-attention` MLA-absorb 的上游预处理，本身不做 softmax/QK/PV。**与 mla_preprocess 的差异**：权重更细粒度（W^DQ/W^UQ/W^QR/W^DKV/W^KR 独立而非 W^DQKV/W^UQ/W^UK），支持 Ascend950（FP8），CV 同步更复杂（6 event）。

## 1. 输入输出

### 输入（def.cpp L23-117；README L62-86）

| 输入 | 必选/可选 | 含义 | dtype | format |
|---|---|---|---|---|
| token_x | 必选 | x，[B,S,He]/[T,He] | BF16/INT8 | ND |
| weight_dq | 必选 | W^DQ 下采样 [He,Hcq] | BF16/INT8 | NZ |
| weight_uq_qr | 必选 | [W^UQ\|W^QR] [Hcq,N*(D+Dr)] | BF16/INT8 | NZ |
| weight_uk | 必选 | W^UK [N,D,Hckv] | BF16 | ND |
| weight_dkv_kr | 必选 | [W^DKV\|W^KR] [He,Hckv+Dr] | BF16/INT8 | NZ |
| rmsnorm_gamma_cq/ckv | 必选 | RmsNorm γ [Hcq]/[Hckv] | BF16 | ND |
| rope_sin/cos | 必选 | RoPE [T,Dr]/[B,S,Dr] | BF16 | ND |
| cache_index | 必选 | Cache 索引 [T]/[B,S] | INT64 | ND |
| kv_cache/kr_cache | 必选 | KVCache（输入/输出同 tensor） | BF16/INT8 | ND |
| dequant_scale_x/w_dq/w_dkv_kr | 可选 | 预留（未使用，传空指针） | FLOAT | ND |
| dequant_scale_w_uq_qr | 可选 | MatmulQcQr 反量化 per-channel | FLOAT | ND |
| quant_scale_ckv/ckr | 可选 | 输出量化参数 | FLOAT | ND |
| smooth_scales_cq | 可选 | RmsNormCq 动态量化参数 | FLOAT | ND |

### 输出（def.cpp L118-133）

| 输出 | 含义 | dtype |
|---|---|---|
| query | q^N [B,S,N,Hckv]/[T,N,Hckv] | BF16 |
| query_rope | q^R [B,S,N,Dr]/[T,N,Dr] | BF16 |
| kv_cache | k^C（原地更新） | BF16/INT8 |
| kr_cache | k^R（原地更新） | BF16/INT8 |

### 属性（def.cpp L134-136）
`rmsnorm_epsilon_cq`(1e-5)、`rmsnorm_epsilon_ckv`(1e-5)、`cache_mode`("BNSD"=PA_BSND/PA_NZ)。

### 约束（README L91-128）
3 场景：非量化(BF16)/部分量化KV非量化(weight_uq_qr=INT8)/部分量化KV量化(kv_cache=INT8 per-channel)。B/S/T/Skv=0 时 query/kv_cache 输出空 tensor。

## 2. 核心计算公式

```
q^N = RmsNorm(x·W^DQ)·W^UQ·W^UK              # 公式(a), 设计介绍 L154
q^R = ROPE(RmsNorm(x·W^DQ)·W^QR)              # 公式(b), L158
k^R = ROPE(x·W^KR)                            # 公式(c), L162
c^KV = RmsNorm(x·W^DKV)                       # 公式(d), L166
```
W^UQ+W^QR 与 W^DKV+W^KR 分别横向拼接为一个 matmul 后 split（公式 7/11）。

## 3. 计算流程（7 阶段 CV 流水，6 sync event）

详见 `artifacts/mla-prolog/pipeline-diagram.md`。主循环（设计介绍 L83-146）：

| # | 阶段 | 单元 | 操作 | 同步 |
|---|---|---|---|---|
| 1 | MatmulCq | AIC | x·W^DQ → cQ | →SYNC_MMCQ_NORMSCQ |
| 2 | MatmulCkvKr | AIC | x·[W^DKV\|W^KR] | →SYNC_MMCKVKR_NORMROPE |
| 3 | RmsNormCq | AIV | γ·cQ/RMS(cQ) | ←等1, →全核→↺Cube |
| 4 | RmsNormRopeScatterCkvKr | AIV | RmsNorm(cKV)+RoPE(kR)+Cache | ←等2 |
| 5 | MatmulQcQr | AIC | cQ_norm·[W^UQ\|W^QR] | ←等3, →SYNC_MMQCQR_ROPEQR, →全核→↺全核 |
| 6 | RopeQr | AIV | RoPE(qR) | ←等5 |
| 7 | MatmulQn | AIC | qC·W^UK → q^N | ←等全核(5) |

## 4. tiling 与 dispatch（详见 `artifacts/mla-prolog/tiling_dispatch_tables.md`）

- **tiling key 二进制**：CACHE_MODE(2bits)+SCENARIO(2bits)+QUANT_MODE(4bits)+DEQUANT/COMPUTE(2bits)+EMPTY(2bits)（设计介绍 L70-79）。
- **8 模板参数**：`CacheMode, Scenario, QuantMode, EnableDequantOpt, EnableGroupComputeOpt, EmptyTensorMode, ActualSeqLenMode, SplitMMode, CvMode`（mla_prolog.cpp L20-22）。
- **3 场景**：NO_QUANT(BF16)/PARTIAL_QUANT_KV_NO_QUANT(INT8_uq_qr)/PARTIAL_QUANT_KV_QUANT_PER_CHANNEL(INT8 cache)（L69-99）。
- **4 matmul 切分**：BS 合轴 M，N 轴按 Cube 核数（MatmulCq Hcq / MatmulCkvKr Hckv+Dr / MatmulQcQr N*(D+Dr) / MatmulQn Hckv）。
- **架构**：A2/A3(BF16/INT8) + 950(额外 FP8_E4M3FN)。`MIX_AIC_1_2`（1:2 CV 配比）。

## 5. 与 mla_preprocess 的对比

| 差异 | mla_prolog (V1) | mla_preprocess (V2+) |
|---|---|---|
| 权重 | W^DQ/W^UQ/W^QR/W^DKV/W^KR 独立 | W^DQKV/W^UQ/W^UK（W^DQKV=W^DQ+W^DKV 合并） |
| 架构支持 | A2/A3 + 950(FP8) | 仅 A2/A3 |
| CV sync | 6 event（更复杂，含全核同步） | ProcessCube+ProcessVector 两阶段 |
| token_x dtype | BF16/INT8 | FP16/BF16 |
| 量化 | per-channel cache 量化 | per_tensor 静态非对称 |
| 公式 | `x·W^DQ→RmsNorm→[W^UQ\|W^QR]` | `x→RmsNormQuant₁→W^DQKV→[W^UQ\|W^UK]` |

两者输出等价（q^N/q^R + k^C/k^R），但权重拆分与量化策略不同。`kernel-mla-preprocess` 页有详细对比。

## 6. 架构支持（README L7-9；def.cpp L145-267）

| 产品 | 架构 | 支持 | dtype |
|---|---|---|---|
| Atlas A2 训练/推理 | Ascend910B1 | √ | BF16/INT8 |
| Atlas A3 训练/推理 | Ascend910_93 | √ | BF16/INT8 |
| Ascend 950PR/950DT | Ascend950 | √ | BF16/INT8/FP8_E4M3FN |

## 7. Agent actions

1. 识别为 MLA prolog V1（权重 W^DQ/W^UQ/W^QR/W^DKV/W^KR 独立），与 mla_preprocess V2+ 区分。
2. 按架构选 A2/A3(BF16/INT8) 或 950(FP8)；按 scenario 选量化分支（NO_QUANT/PARTIAL_QUANT）。
3. 4 matmul：W^UQ+W^QR 与 W^DKV+W^KR 分别合并 matmul 后 split；MatmulQn 需等全核 MatmulQcQr。
4. 6 CV sync event（CANN 专有）抽象为 tile 依赖链：Cube matmul → Vector norm → Cube matmul → Vector rope。
5. Cache scatter 写回按 cache_index；PA 场景支持 ND/NZ 格式。
6. 参考 `artifacts/mla-prolog/mla-prolog-triton-skeleton.py`（pseudocode）。

## 8. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、FP8 精度边界、MLA 权重吸收实现、预留参数（未使用）均未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 `ops-transformer-mla-prolog`。

<!-- okf:related:start -->

# 相关

- 实践案例: [融合算子综合页](../reference/op-class-fused_ops.md) — 主题相关：共享 attention, fusion, rmsnorm, rope
- 实践案例: [MatMul / GEMM 高性能算子](../reference/op-class-matmul.md) — 主题相关：共享 matmul
- 实践案例: [归一化算子优化](../reference/op-class-normalization.md) — 主题相关：共享 rmsnorm
- 实践案例: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 rmsnorm
- 实践案例: [matmul core mapping 策略选择（列/行/对角）](../runbooks/compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 实践案例: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../runbooks/compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 实践案例: [计算优化: fixpipe 随路量化 + K 维循环展开 + 预取](../runbooks/compute-pipeline/exp_matmul_002.md) — 主题相关：共享 matmul
- 实践案例: [算子融合模式与约束决策](../runbooks/compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 fusion, rmsnorm, rope
- 实践案例: [pad slot 单次检查提前退出，跳过无效填充序列](../runbooks/compute-pipeline/pad-slot-early-exit.md) — 主题相关：共享 kv_cache
- 实践案例: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../runbooks/memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 实践案例: [Memory Access Optimization: fp8 value casting](../runbooks/memory-sram/exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 attention
- 实践案例: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](../runbooks/memory-sram/exp_fused_rms_norm_001.md) — 同属优化知识，主题相关
- 实践案例: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](../runbooks/memory-sram/exp_gemma_rmsnorm_001.md) — 主题相关：共享 matmul
- 实践案例: [访存优化：大尺寸 matmul 的 256 分块与 L0C 复用](../runbooks/memory-sram/exp_matmul_001.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块](../runbooks/memory-sram/exp_matmul_003.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块](../runbooks/memory-sram/exp_matmul_004.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：双转置矩阵乘的 512B 对齐分块](../runbooks/memory-sram/exp_matmul_005.md) — 主题相关：共享 matmul
- 实践案例: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](../runbooks/memory-sram/exp_rmsnorm_fused_quant_001.md) — 同属优化知识，主题相关
- 实践案例: [Cube-Vector 场景行宽必须 512B 对齐](../runbooks/memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 实践案例: [tl.dot vs torch.matmul 精度差异诊断](../runbooks/meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../runbooks/meta-skills/quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Rmsnorm Fused Skeleton](../reference/skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 ascend910b1, ascend950pr, fp32_accumulation, tail_mask
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, matmul, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, kv_cache, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, kv_cache, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, kv_cache, l0c_buffer, mla, quant, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — MLA prolog/preprocess 算子家族
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — MLA prolog 的版本迭代
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, quant, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, quant, unified_buffer
- 相关主题: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](moe-init-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, quant, unified_buffer
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, kv_cache, tail_mask, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, kv_cache, l0c_buffer, l0c_to_ub, l1_buffer, mla, quant, tail_mask, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, fp8, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, fp32_accumulation, rmsnorm, rope, unified_buffer
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — 主题相关：共享 ascend910b1, rope, tail_mask
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, kv_cache, l0c_buffer, l0c_to_ub, l1_buffer, mla, tail_mask, unified_buffer

<!-- okf:related:end -->
