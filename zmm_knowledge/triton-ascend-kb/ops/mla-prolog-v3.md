---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MlaPrologV3 (MLA pre-process) on Ascend NPU
description: CANN ops-transformer 中的 MlaPrologV3 算子是 DeepSeek V2/V3 核心创新 MLA（Multi-head Latent Attention） 的前处理融合算子。对输入 tokenx 执行 4 路 matmul（下采样 + 拼接上采样 + 拼接下采样 + 上采样）+ 2 个 RmsNorm + 2 个 RoPE 位置编码 + KVCache/KRCache 更新，输出 query（q^N）、queryrope（q^Rrope）、...
tags: [mla_prolog, mla, deepseek, attention, prefill, rmsnorm, rope, matmul, cache, fp32_accumulation, unified_buffer, l0c_buffer, l1_buffer, l0a_buffer, l0b_buffer, cube_vector_1_to_2, ascend910b1, ascend910b2, ascend910_93, ascend950pr, ascend950dt, mla_prolog_v3]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/mla_prolog_v3/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/mla_prolog_v3/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/mla-prolog-v3.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/mla_prolog_v3/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/mla_prolog_v3/)。

# MlaPrologV3 (MLA pre-process)

CANN ops-transformer 中的 `MlaPrologV3` 算子是 **DeepSeek V2/V3 核心创新 MLA（Multi-head Latent Attention）** 的前处理融合算子。对输入 token_x 执行 4 路 matmul（下采样 + 拼接上采样 + 拼接下采样 + 上采样）+ 2 个 RmsNorm + 2 个 RoPE 位置编码 + KVCache/KRCache 更新，输出 query（q^N）、query_rope（q^R_rope）、kv_cache、kr_cache。三架构（arch32 A2/A3 + arch35 950）。

## 1. 输入输出

### 输入（按 `mla_prolog_v3_def.cpp` 与 README L80-116）

| 输入 | 必选/可选 | 含义 | dtype | 格式 |
|------|-----------|------|-------|------|
| token_x | 必选 | 输入 token | BF16/INT8/FP8_E4M3FN | ND |
| weight_dq | 必选 | 下采样 W^{DQ} [He, Hcq] | BF16/INT8/FP8_E4M3FN | **FRACTAL_NZ** |
| weight_uq_qr | 必选 | [W^{UQ}\|W^{QR}] 拼接 [Hcq, Hckv+Dr] | BF16/INT8/FP8_E4M3FN | **FRACTAL_NZ** |
| weight_uk | 必选 | 上采样 W^{UK} [Hckv, N*Hckv] | BF16/FP16 | ND |
| weight_dkv_kr | 必选 | [W^{DKV}\|W^{KR}] 拼接 [He, Hckv+Dr] | BF16/INT8/FP8_E4M3FN | **FRACTAL_NZ** |
| rmsnorm_gamma_cq/ckv | 必选 | RmsNorm γ 参数 | BF16/FP16 | ND |
| rope_sin/cos | 必选 | RoPE sin/cos 参数 | BF16/FP16 | ND |
| kv_cache | 输入/输出 | KVCache | FP16/BF16/INT8 | ND |
| kr_cache | 输入/输出 | KRCache | FP16/BF16/INT8 | ND |
| cache_index | 必选 | cache 更新索引 [B,S] 或 [T,] | INT64 | ND |
| 可选 20+ | 可选 | dequant_scale_x/dq/uq_qr/dkv_kr、quant_scale_ckv/ckr、smooth_scales_cq、actual_seq_len、kNopeClipAlpha、qcQrScale、kcScale 等 | 按模式 | ND |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| query (q^N) | 上采样后的 query [T,N,Hckv] | FP16/BF16/INT8 |
| query_rope (q^R_rope) | RoPE 后的 query [T,N,Dr] | FP16/BF16/INT8 |
| kv_cache（原地更新） | c^{KV}_norm 写入 cache | FP16/BF16/INT8 |
| kr_cache（原地更新） | k^R_rope 写入 cache | FP16/BF16/INT8 |
| dequant_scale_q_nope | query 量化参数（可选） | FLOAT |
| query_norm | c^Q（可选） | INT8/BF16 |
| dequant_scale_q_norm | query_norm 量化参数（可选） | FLOAT |

### 属性（V3）

`cache_mode`（PA_BSND/PA_NZ）、`qc_qr_scale`（α_q 默认1.0）、`kc_scale`（α_kv 默认1.0）、`query_norm_flag`、`weight_quant_mode`、`kv_cache_quant_mode`、`query_quant_mode`、`ckvkr_repo_mode`、`quant_scale_repo_mode`、`tile_size`（per-tile 128）、`rmsnorm_epsilon_cq/ckv`。

## 2. 核心计算公式

```
c^Q  = α_q · RmsNorm(x · W^{DQ})                            # ① 下采样 + RmsNorm
q^N  = c^Q · W^{UQ} · W^{UK}                                # ② 上采样 -> query
q^R  = ROPE(c^Q · W^{QR})                                   # ③ RoPE -> query_rope
c^{KV} = α_kv · RmsNorm(x · W^{DKV})                       # ④ 下采样 + RmsNorm -> kv_cache
k^R  = ROPE(x · W^{KR})                                     # ⑤ RoPE -> kr_cache
```

利用矩阵乘拼接优化：W^{UQ}+W^{QR} 拼为 [W^{UQ}|W^{QR}]，W^{DKV}+W^{KR} 拼为 [W^{DKV}|W^{KR}]，单次 matmul 后 split。

## 3. 计算流程

### 3.1 四路 matmul + 2 RmsNorm + 2 RoPE（设计介绍 L15-25）

```
x → MatmulCq (x·W^{DQ}) → c^Q → RmsNormCq → c_norm^Q
                                            → MatmulQcQr ([W^{UQ}|W^{QR}]) → [q^C|q^R] → split
                                                                                  q^C → MatmulQn (W^{UK}) → q^N (query)
                                                                                  q^R → RoPE → q^R_rope (query_rope)
x → MatmulCkvKr ([W^{DKV}|W^{KR}]) → [c^{KV}|k^R] → split
                                          c^{KV} → RmsNormCkv → kv_cache
                                          k^R    → RoPE → kr_cache
```

### 3.2 CV 流水 6 同步点（设计介绍 L47-68）

不同 matmul 的分核策略不同（Hcq 按 AIC 核数、Hckv 固定 64 块、Hqc 按 AIC 核数），导致 Cube 和 Vector 之间需要 6 个核间同步点（SYNC_MMCQ_NORMSCQ / SYNC_MMCKVKR_NORMROPE / SYNC_MMQCQR_ROPEQR / SYNC_ALL_CUBE / SYNC_ALL_VECTOR 等）。这是 CANN 专有机制，不迁移到 Triton。

### 3.3 TilingKey（26 位二进制，9 字段）

`mla_prolog_v3.cpp` L25-27 的 9 个 uint8_t 模板参数对应 TilingKey（设计介绍 L70-85）：

| 字段 | 位 | 说明 |
|------|----|------|
| CACHE_MODE | 0-3 | 0:BNSD / 1:PA_BSND / 2:PA_NZ |
| SCENARIO | 4-5 | 0:FP16 / 1:非量化 / 2:量化 |
| QUANT_MODE | 6-9 | 8 种量化组合（非量化/MMQcQr/KVcache pertile/全量/Mxfp8 等） |
| ENABLE_DEQUANT_OPT | 10 | 反量化使能 |
| ENABLE_GROUP_COMPUTE_OPT | 11 | 算力分组优化 |
| EMPTY_TENSOR_MODE | 12-13 | 空 tensor（KV 空/Query 空） |
| ACTUAL_SEQ_LEN_MODE | 14-15 | actualSeqLen 使能 |
| SPLIT_M_MODE | 16-17 | 0:切N / 1:切M |
| CV_MODE | 18-25 | CV 配比（1:1 / 1:2） |

## 4. 架构与产品支持

| 产品 | 架构 | config（def.cpp） | kernel 入口 |
|------|------|-------------------|-------------|
| Atlas A2 训练/推理 | arch32（910B） | `ascend910b` L171 | `mla_prolog_v3.cpp`（共享 mla_prolog/） |
| Atlas A3 训练/推理 | ascend910_93 | `ascend910_93` L172 | 同上 |
| Ascend 950PR/950DT | arch35 | `ascend950` L315 | 同上（arch35 单独 dtype/layout） |

三架构均支持，无 310P。kernel 实现共享 `mla_prolog/op_kernel/kernel_mla_prolog_split_n.h`（V3 与 V1/V2 共享核心实现）。

## 5. 量化模式

支持 8 种量化模式（QUANT_MODE 0-8，README L147-203）：

| 模式 | token_x | weight_dq/uq_qr/dkv_kr | kv_cache | query 输出 |
|------|---------|------------------------|----------|-----------|
| 非量化 | BF16 | BF16 | FP16/BF16 | FP16/BF16 |
| 部分量化 | BF16 | pertoken INT8 | 非量化/pertensor/pertile | FP16/BF16 |
| 全量化 | pertoken INT8 | perchannel INT8 | 非量化/pertensor/pertile | pertoken_head INT8 |
| Mxfp8 | FP8_E4M3FN | FP8_E4M3FN | pertensor | pertoken_head |

## 6. Agent actions

1. 确认目标架构（arch32 / arch35）与量化模式（QUANT_MODE 0-8）。
2. 4 路 matmul 映射：`tl.dot` × 4（下采样 + 2 个拼接 matmul + 上采样）。
3. 2 个 RmsNorm：`x * rsqrt(mean(x^2) + eps) * gamma`。
4. 2 个 RoPE：偶奇拆分 * cos + 翻转 * sin -> 合并。
5. KVCache/KRCache scatter：`tl.store(cache + cache_index, ...)`。
6. V3 特有：α_q/α_kv 尺度矫正因子（qcQrScale/kcScale）、query_norm_flag、per-tile 量化（tileSize=128）。
7. CANN 专有（不迁移）：CV 6 同步点、FRACTAL_NZ 物理格式、TilingKey 26 位 key。
8. Triton 实现参考 `artifacts/mla-prolog-v3/mla-prolog-v3-triton-skeleton.py`（pseudocode）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。kernel 实现来自共享 `mla_prolog/` 目录（V3 入口仅定义模板参数），核心实现细节需以 `kernel_mla_prolog_split_n.h` 为准。tiling 参数数值、per-tile 量化精度边界、cacheIndex 与 cache 更新关系、Mxfp8 支持需结合源码/硬件进一步验证。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 `ops-transformer-mla-prolog-v3`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Legacy 缓存读写协议（triton-knowledge-retrieval）](../reference/SPEC-Retrieve-legacy.md) — 同属优化知识，主题相关
- 实践案例: [融合算子综合页](../reference/op-class-fused_ops.md) — 主题相关：共享 attention, rmsnorm, rope
- 实践案例: [MatMul / GEMM 高性能算子](../reference/op-class-matmul.md) — 主题相关：共享 matmul
- 实践案例: [归一化算子优化](../reference/op-class-normalization.md) — 主题相关：共享 rmsnorm
- 实践案例: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 rmsnorm
- 实践案例: [matmul core mapping 策略选择（列/行/对角）](../runbooks/compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 实践案例: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../runbooks/compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 实践案例: [计算优化: fixpipe 随路量化 + K 维循环展开 + 预取](../runbooks/compute-pipeline/exp_matmul_002.md) — 主题相关：共享 matmul
- 实践案例: [算子融合模式与约束决策](../runbooks/compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 rmsnorm, rope
- 实践案例: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../runbooks/memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 实践案例: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](../runbooks/memory-sram/exp_fused_rms_norm_001.md) — 同属优化知识，主题相关
- 实践案例: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](../runbooks/memory-sram/exp_gemma_rmsnorm_001.md) — 主题相关：共享 matmul
- 实践案例: [访存优化：大尺寸 matmul 的 256 分块与 L0C 复用](../runbooks/memory-sram/exp_matmul_001.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块](../runbooks/memory-sram/exp_matmul_003.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块](../runbooks/memory-sram/exp_matmul_004.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：双转置矩阵乘的 512B 对齐分块](../runbooks/memory-sram/exp_matmul_005.md) — 主题相关：共享 matmul
- 实践案例: [Memory Access Optimization: paged kv cache](../runbooks/memory-sram/exp_mla_paged_attn_001.md) — 主题相关：共享 attention
- 实践案例: [Memory Access Optimization: paged kv cache with gqa head tiling](../runbooks/memory-sram/exp_paged_gqa_001.md) — 主题相关：共享 attention
- 实践案例: [Cube-Vector 场景行宽必须 512B 对齐](../runbooks/memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 实践案例: [tl.dot vs torch.matmul 精度差异诊断](../runbooks/meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 实践案例: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](../runbooks/portability/exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 attention
- 相关主题: [Rmsnorm Fused Skeleton](../reference/skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 ascend910b1, ascend950dt, ascend950pr, fp32_accumulation
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, matmul, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, matmul, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, matmul, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, matmul, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, mla, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](mla-preprocess.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, matmul, mla, rmsnorm, rope, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — MLA prolog 的版本迭代
- 相关主题: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](moe-compute-expert-tokens.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](moe-finalize-routing-v2-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](moe-finalize-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](moe-gating-top-k-softmax-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRoutingV2 (MoE routing v2) on Ascend NPU](moe-init-routing-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeInitRouting (MoE routing) on Ascend NPU](moe-init-routing.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](moe-token-permute-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](moe-token-permute-with-routing-map.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, ascend950dt, ascend950pr, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [NsaCompressGrad (NSA compress backward) on Ascend NPU](nsa-compress-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend910b2, attention, fp32_accumulation, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0a_buffer, l0b_buffer, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, mla, prefill, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, ascend950pr, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, matmul, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, fp32_accumulation, rmsnorm, rope, unified_buffer
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — 主题相关：共享 ascend910b1, rope
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, mla, unified_buffer

<!-- okf:related:end -->
