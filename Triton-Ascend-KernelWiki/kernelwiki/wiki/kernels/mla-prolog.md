---
id: kernel-mla-prolog
title: MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU
type: kernel
tags: [mla, attention, rmsnorm, matmul, rope, kv_cache, fusion, cube_vector_pipe, quant, fp8]
techniques: [fp32_accumulation, tail_mask]
hardware_features: [unified_buffer, l0c_to_ub, cube_vector_1_to_2, l1_buffer, l0c_buffer]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR]
kernel_types: [attention]
operators: [mla_prolog]
symptoms: [precision_mismatch, performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/mla-prolog
related: [kernel-mla-preprocess, kernel-sparse-flash-attention, kernel-prompt-flash-attention, kernel-flash-attention-score]
sources: [ops-transformer-mla-prolog]
performance_claims: []
---

# MlaProlog（MLA 前处理 Prolog V1）

CANN ops-transformer 的 `MlaProlog` 是**推理场景** MLA 前处理融合算子 V1（与 [[kernel-mla-preprocess]] V2+ 不同）。四路计算：x→×W^DQ(RmsNorm)→[×W^UQ·W^UK→q^N]+[×W^QR→RoPE→q^R]+[×W^DKV→RmsNorm→cache→k^C]+[×W^KR→RoPE→cache→k^R]。它是 [[kernel-sparse-flash-attention]]/[[kernel-prompt-flash-attention]] MLA-absorb 的上游预处理，本身不做 softmax/QK/PV。**与 mla_preprocess 的差异**：权重更细粒度（W^DQ/W^UQ/W^QR/W^DKV/W^KR 独立而非 W^DQKV/W^UQ/W^UK），支持 Ascend950（FP8），CV 同步更复杂（6 event）。

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

两者输出等价（q^N/q^R + k^C/k^R），但权重拆分与量化策略不同。[[kernel-mla-preprocess]] 页有详细对比。

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

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、FP8 精度边界、MLA 权重吸收实现、预留参数（未使用）均未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`。来源见 [[ops-transformer-mla-prolog]]。