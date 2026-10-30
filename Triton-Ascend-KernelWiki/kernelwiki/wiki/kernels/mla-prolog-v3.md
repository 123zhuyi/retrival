---
id: kernel-mla-prolog-v3
title: MlaPrologV3 (MLA pre-process) on Ascend NPU
type: kernel
tags: [mla_prolog, mla, deepseek, attention, prefill, rmsnorm, rope, matmul, cache]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, l0c_buffer, l1_buffer, l0a_buffer, l0b_buffer, cube_vector_1_to_2]
architectures: [Ascend910B1, Ascend910B2, Ascend910_93, Ascend950PR, Ascend950DT]
kernel_types: [attention]
operators: [mla_prolog_v3]
symptoms: [precision_mismatch, ub_overflow, performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/mla-prolog-v3
related: [technique-flash-attention-tiling, technique-flash-attention-pipeline, kernel-flash-attention-score]
sources: [ops-transformer-mla-prolog-v3]
performance_claims: []
---

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

本页结论基于源码静态分析（confidence: inferred）。kernel 实现来自共享 `mla_prolog/` 目录（V3 入口仅定义模板参数），核心实现细节需以 `kernel_mla_prolog_split_n.h` 为准。tiling 参数数值、per-tile 量化精度边界、cacheIndex 与 cache 更新关系、Mxfp8 支持需结合源码/硬件进一步验证。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 [[ops-transformer-mla-prolog-v3]]。