---
id: kernel-mla-preprocess
title: MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU
type: kernel
tags: [mla, attention, rmsnorm, quant, matmul, rope, kv_cache, fusion, cube_vector_pipe]
techniques: [fp32_accumulation, tail_mask, contiguous_access]
hardware_features: [unified_buffer, l0c_to_ub, cube_vector_1_to_2, l1_buffer, l0c_buffer]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [attention]
operators: [mla_preprocess]
symptoms: [precision_mismatch, performance_bottleneck, ub_overflow]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/mla-preprocess
related: [kernel-sparse-flash-attention, kernel-prompt-flash-attention, kernel-flash-attention-score]
sources: [ops-transformer-mla-preprocess]
performance_claims: []
---

# MlaPreprocess（MLA 前处理融合算子）

CANN ops-transformer 的 `MlaPreprocess` 是**推理场景** Multi-Head Latent Attention（MLA）的前处理融合算子：把输入 x 经 RmsNormQuant + 降维矩阵乘（W^DQKV）拆分为 Q/K 两路，再各自做 RmsNorm/上采样矩阵乘（W^UQ/W^UK）/RoPE，输出 q^N（nope）、q^R（rope）并把 k^N/k^R 写入 KV cache（README L12-19）。它是 [[kernel-sparse-flash-attention]]/[[kernel-prompt-flash-attention]] MLA 模式（attention_mode=2）的上游预处理，本身不做 softmax/QK/PV。

## 1. 输入输出

### 输入（def.cpp L73-96；README L74-241，多为 REQUIRED）

| 输入 | 含义 | dtype | format |
|---|---|---|---|
| input | x，[tokenNum, hiddenSize] | FP16/BF16 | ND |
| gamma_0/beta_0/quant_scale_0 | 1st RmsNormQuant 参数 | FP16/BF16 | ND |
| quant_offset_0 | 1st 量化偏移 | INT8 | NZ |
| wdqkv | 降维矩阵 W^DQKV，[2112,hiddenSize] | INT8/BF16 | NZ |
| descale_0/bias_0 | mm1 系数/偏置 | INT64/FLOAT / INT32 | ND |
| gamma_1/beta_1/quant_scale_1/quant_offset_1 | 2nd RmsNormQuant 参数 | FP16/BF16 / INT8 | ND |
| wuq | 上采样 Q 矩阵 W^UQ，[headNum*192,1536] | INT8/BF16 | NZ |
| descale_1/bias_1 | mm2 系数/偏置 | INT64/FLOAT / INT32 | ND/NZ |
| gamma_2 | 3rd RmsNorm（K 路）参数，[512] | FP16/BF16 | ND |
| cos/sin | RoPE 正余弦，[tokenNum,64] | INT8 | NZ |
| wuk | 上采样 K 矩阵 W^UK | FP16/BF16 | ND/NZ |
| kv_cache / kv_cache_rope | KV cache（与输出同 tensor） | FP16/BF16/INT8 | ND/NZ（随 cacheMode） |
| slot_mapping | cache 索引，[tokenNum] | INT32 | ND |
| ctkv_scale / q_nope_scale | 输出量化系数（cacheMode=2） | FP16/BF16 | ND |

### 输出（def.cpp L97-100；README L334-360）

| 输出 | 含义 | dtype | format |
|---|---|---|---|
| q_out | q^N（nope），[tokenNum,headNum,512]（cacheMode 0 为 576 拼接） | FP16/BF16/INT8 | ND |
| kv_cache_out | k^N 写回 cache | FP16/BF16/INT8 | ND/NZ |
| q_rope_out | q^R（rope），[tokenNum,headNum,64] | FP16/BF16 | ND |
| kr_cache_out | k^R 写回 cache | FP16/BF16 | ND/NZ |

### 属性（def.cpp L101-113）
- `wdq_dim`(128,预留)、`q_rope_dim`/`k_rope_dim`(64,预留)、`epsilon`(1e-5)、`q_rotary_coeff`/`k_rotary_coeff`(2,预留)。
- `transepose_wdq`/`wuq`/`wuk`(true,预留)、`do_rms_norm`(true,预留)、`wdkv_split_count`(1,预留)。
- `cache_mode`(0-3)、`quant_mode`(0-3，仅 0=per_tensor 静态非对称实现，1/2/3 未实现，README L315)。

### 约束（README L363-371）
tokenNum 0-256；hiddenSize 2048-10240（256 倍数）；headNum {16,32,64,128}；blockNum 192；blockSize 128。BF16 权重 -> input BF16 且 hiddenSize 仅 6144。

## 2. 核心计算公式

```
RMS(x) = sqrt(mean(x²) + eps);   RmsNorm(x) = γ · x / RMS(x)
RmsNormQuant(x) = (RmsNorm(x) + bias) · deqScale
q^N = RmsNormQuant(x) · W^DQKV · W^UK      # 数学 2-matmul 形式（MLA 权重吸收 W^UQ·W^UK 折叠）
q^R = ROPE(x^Q)
k^N = Cache(RmsNorm(RmsNormQuant(x)))
k^R = Cache(ROPE(RmsNormQuant(x)))
```
（README L20-54）。kernel 实际实现 3 个 matmul（mm1=W^DQKV, mm2=W^UQ, mm3=W^UK，tilingdata L51-53）；是否运行时做权重吸收 `needs_verification`。

## 3. 计算流程（CV 分离）

kernel 入口先 `ProcessCube()` 再 `ProcessVector()`（mla_preprocess.cpp L52-53）：

| 阶段 | 单元 | 操作 | tiling 方法 |
|---|---|---|---|
| Cube | AIC | 3 matmul（mm1 降维 / mm2 上采样Q / mm3 上采样K）+ RoPEConcat | `RopeConcatTiling`（tiling.h L54） |
| Vector | AIV | 2× RmsNormQuant + RmsNorm₃ + EinSumQuant + Cache 写回 | `RmsNormQuantTiling` / `EinSumQuantTiling`（L53/55） |

数据流：x -> RmsNormQuant₁ -> mm1 -> 拆 Q/K；Q: RmsNormQuant₂ -> mm2 -> 拆 nope(rope)/rope -> nope: mm3 -> q^N，rope: RoPE -> q^R；K: RmsNorm₃ -> cache k^N，RoPE -> cache k^R。

## 4. tiling 与 dispatch（概要，详见 artifact `tiling_dispatch_tables.md`）

- dtype dispatch：FP16 -> `_fp16.h`；BF16 -> `_bf16.h` + `_no_quant.h`（mla_preprocess.cpp L17-22）。
- tiling key = cacheMode × 3 输出格式(ND/NZ)（L46+，步长 4，8 组合/cacheMode）。
- 核间切分：tokenNum 沿 numCore，每核 perTaskNum、尾核 resTaskNum（tilingdata L47-50）。
- 3 matmul 基本块 m0/k0/n0 + loop + swizzle + ping-pong（MlaPpMatmulTilingData L25-43）。
- workspace 3 等大 section（s1/s2/s3，L41-43）。
- buffer 搬运由 9 个 `*_iterator.h`（GM/L1/L0C/UB 间）封装，CANN 专有。

## 5. 与 MLA attention 的关系

`mla_preprocess` 是 MLA attention 的**上游预处理**：
- 输出 q^N(512)+q^R(64) 与 k^N(512)+k^R(64) 供下游 [[kernel-sparse-flash-attention]]/[[kernel-prompt-flash-attention]] 的 MLA-absorb（attention_mode=2）使用。
- 下游 attention 把 nope+rope 沿 D 拼接成 576 做 QK；key/value 共享张量。
- 本算子**不做** softmax/QK/PV，只做 norm+matmul+rope+cache 融合。

## 6. 架构支持（README L7-8；def.cpp L118-119）

| 产品 | 架构 | 支持 |
|---|---|---|
| Atlas A2 训练/推理 | Ascend910B1 | √ |
| Atlas A3 训练/推理 | Ascend910_93 | √ |

不支持 950/310P。FP16/BF16（权重 INT8 量化）。

## 7. Agent actions

1. 识别为 MLA 前处理融合（norm+matmul+rope+cache），非 attention 主算子；下游接 MLA attention。
2. 按 dtype 选 FP16/BF16/no_quant 模板；按 cacheMode 选输出格式(ND/NZ)与量化分支。
3. Cube 侧 3 个 matmul（mm1/mm2/mm3）注意 MLA 权重吸收等价性（数学 2-matmul）。
4. Vector 侧 RmsNormQuant 用 FP32 reduce；quantMode 仅 0 实现。
5. Cache 写回按 slot_mapping scatter store；cacheMode 决定 cache shape/格式/int8 量化。
6. Triton 实现参考 `artifacts/mla-preprocess/mla-preprocess-triton-skeleton.py`（pseudocode）。
7. CANN 专有（buffer iterator/ND↔NZ/CV 分离/swizzle）不迁移，抽象为 fused load/store + 单 program。

## 8. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、workspace 预算、MLA 权重吸收实现、quantMode 1/2/3（未实现）、cacheMode=2 int8 量化边界均未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 [[ops-transformer-mla-preprocess]]。
