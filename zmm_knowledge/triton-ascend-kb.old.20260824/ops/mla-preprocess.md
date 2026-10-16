---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU
description: CANN ops-transformer 的 MlaPreprocess 是推理场景 Multi-Head Latent Attention（MLA）的前处理融合算子：把输入 x 经 RmsNormQuant + 降维矩阵乘（W^DQKV）拆分为 Q/K 两路，再各自做 RmsNorm/上采样矩阵乘（W^UQ/W^UK）/RoPE，输出 q^N（nope）、q^R（rope）并把 k^N/k^R 写入 KV cache（README L12-19）。它是 kernel-...
tags: [mla, attention, rmsnorm, quant, matmul, rope, kv_cache, fusion, cube_vector_pipe, fp32_accumulation, tail_mask, contiguous_access, unified_buffer, l0c_to_ub, cube_vector_1_to_2, l1_buffer, l0c_buffer, ascend910b1, ascend910_93, mla_preprocess]
confidence: inferred
status: active
resource: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/mla_preprocess/
created_at: '2026-08-06T07:12:24Z'
updated_at: '2026-08-06T07:12:24Z'
---

sources:
  - url: https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/mla_preprocess/
    role: primary

> 迁移备注：本卡由 A 的 `kernelwiki/wiki/kernels/mla-preprocess.md` 迁移而来，原始为 CANN 源码静态分析，status: active / confidence: inferred，未硬件实测。资源 pinned 到 [https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/mla_preprocess/](https://gitcode.com/cann/ops-transformer/tree/d28d8f99a8858b71caa27a535da32bb5ba68ce2a/attention/mla_preprocess/)。

# MlaPreprocess（MLA 前处理融合算子）

CANN ops-transformer 的 `MlaPreprocess` 是**推理场景** Multi-Head Latent Attention（MLA）的前处理融合算子：把输入 x 经 RmsNormQuant + 降维矩阵乘（W^DQKV）拆分为 Q/K 两路，再各自做 RmsNorm/上采样矩阵乘（W^UQ/W^UK）/RoPE，输出 q^N（nope）、q^R（rope）并把 k^N/k^R 写入 KV cache（README L12-19）。它是 `kernel-sparse-flash-attention`/`kernel-prompt-flash-attention` MLA 模式（attention_mode=2）的上游预处理，本身不做 softmax/QK/PV。

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
- 输出 q^N(512)+q^R(64) 与 k^N(512)+k^R(64) 供下游 `kernel-sparse-flash-attention`/`kernel-prompt-flash-attention` 的 MLA-absorb（attention_mode=2）使用。
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

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、workspace 预算、MLA 权重吸收实现、quantMode 1/2/3（未实现）、cacheMode=2 int8 量化边界均未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 `ops-transformer-mla-preprocess`。

<!-- okf:related:start -->

# 相关

- 实践案例: [Elementwise 算子优化](../reference/op-class-elementwise.md) — 主题相关：共享 fusion
- 实践案例: [融合算子综合页](../reference/op-class-fused_ops.md) — 主题相关：共享 attention, fusion, rmsnorm, rope
- 实践案例: [MatMul / GEMM 高性能算子](../reference/op-class-matmul.md) — 主题相关：共享 matmul
- 实践案例: [归一化算子优化](../reference/op-class-normalization.md) — 主题相关：共享 rmsnorm
- 实践案例: [RoPE（旋转位置编码）算子优化](../reference/op-class-rope.md) — 主题相关：共享 rope
- 实践案例: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 主题相关：共享 rmsnorm
- 实践案例: [dl.async_task 实现 Cube/Vector 异构并行](../runbooks/compute-pipeline/async-task-cube-vector.md) — 主题相关：共享 fusion
- 实践案例: [matmul core mapping 策略选择（列/行/对角）](../runbooks/compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 实践案例: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../runbooks/compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 实践案例: [算子融合模式与约束决策](../runbooks/compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 fusion, rmsnorm, rope
- 实践案例: [pad slot 单次检查提前退出，跳过无效填充序列](../runbooks/compute-pipeline/pad-slot-early-exit.md) — 主题相关：共享 kv_cache
- 实践案例: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../runbooks/memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 实践案例: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](../runbooks/memory-sram/parallel-axis-align.md) — 主题相关：共享 contiguous_access
- 实践案例: [Cube-Vector 场景行宽必须 512B 对齐](../runbooks/memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 实践案例: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../runbooks/meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 attention
- 实践案例: [tl.dot vs torch.matmul 精度差异诊断](../runbooks/meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../runbooks/meta-skills/quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Rmsnorm Fused Skeleton](../reference/skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [Normalization Compute 优化聚合](../runbooks/algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 rmsnorm
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../runbooks/algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 rmsnorm
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](../runbooks/compute-pipeline/instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 matmul
- 相关主题: [Matmul Compute 优化聚合](../runbooks/compute-pipeline/traj-matmul-compute.md) — 主题相关：共享 matmul
- 相关主题: [Matmul Scheduling 优化聚合](../runbooks/compute-pipeline/traj-matmul-scheduling.md) — 主题相关：共享 matmul
- 相关主题: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](../runbooks/compute-pipeline/traj-matmul-split.md) — 主题相关：共享 matmul
- 相关主题: [Normalization Scheduling 优化聚合](../runbooks/compute-pipeline/traj-normalization-scheduling.md) — 主题相关：共享 rmsnorm
- 相关主题: [Reduction Compute 优化聚合](../runbooks/compute-pipeline/traj-reduction-compute.md) — 主题相关：共享 rmsnorm
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](../runbooks/compute-pipeline/traj-rope-compute.md) — 主题相关：共享 rope
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](../runbooks/compute-pipeline/traj-rope-scheduling.md) — 主题相关：共享 kv_cache, rope
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](../runbooks/memory-sram/traj-matmul-memory_access.md) — 主题相关：共享 matmul, rmsnorm
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](../runbooks/memory-sram/traj-rope-memory_access.md) — 主题相关：共享 rope
- 相关主题: [Matmul 硬件适配优化聚合](../runbooks/portability/traj-matmul-hardware_adaptation.md) — 主题相关：共享 matmul
- 相关主题: [Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB Tiling、FMA 与交错复数布局](../runbooks/portability/traj-rope-hardware_adaptation.md) — 主题相关：共享 rope
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](block-sparse-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](block-sparse-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 ascend910b1, contiguous_access, fp32_accumulation, tail_mask
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 ascend910b1, fp32_accumulation, tail_mask
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [FlashAttentionScore training-forward on Ascend NPU](flash-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](fused-floyd-attention-grad.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, matmul, unified_buffer
- 相关主题: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](fused-floyd-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](fused-infer-attention-score.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, kv_cache, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [IncreFlashAttention inference-decode on Ascend NPU](incre-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, kv_cache, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](kv-quant-sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, kv_cache, l0c_buffer, mla, quant, unified_buffer
- 相关主题: [LightningIndexer (sparse attention indexer) on Ascend NPU](lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, unified_buffer
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](mla-prolog-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, matmul, mla, rmsnorm, rope, unified_buffer
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](mla-prolog.md) — MLA prolog/preprocess 算子家族
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, quant, unified_buffer
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, quant, unified_buffer
- 相关主题: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](moe-init-routing-v3.md) — 主题相关：共享 ascend910_93, ascend910b1, fp32_accumulation, quant, unified_buffer
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](nsa-compress-attention-infer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](nsa-compress-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, contiguous_access, fp32_accumulation, kv_cache, tail_mask, unified_buffer
- 相关主题: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](nsa-selected-attention.md) — 主题相关：共享 ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l1_buffer, tail_mask, unified_buffer
- 相关主题: [PromptFlashAttention inference/prefill on Ascend NPU](prompt-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, kv_cache, l0c_buffer, l0c_to_ub, l1_buffer, mla, quant, tail_mask, unified_buffer
- 相关主题: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](quant-lightning-indexer.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, tail_mask, unified_buffer
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](rain-fusion-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, l0c_buffer, l0c_to_ub, l1_buffer, matmul, tail_mask, unified_buffer
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, fp32_accumulation, matmul, unified_buffer
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 ascend910b1, contiguous_access, fp32_accumulation, rmsnorm, rope, unified_buffer
- 相关主题: [RoPE paired-channel pointer addressing](rope-pointer-addressing.md) — 主题相关：共享 ascend910b1, rope, tail_mask
- 相关主题: [SparseFlashAttention inference (sparse KV) on Ascend NPU](sparse-flash-attention.md) — 主题相关：共享 ascend910_93, ascend910b1, attention, cube_vector_1_to_2, cube_vector_pipe, fp32_accumulation, kv_cache, l0c_buffer, l0c_to_ub, l1_buffer, mla, tail_mask, unified_buffer

<!-- okf:related:end -->
