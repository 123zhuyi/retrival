---
id: kernel-incre-flash-attention
title: IncreFlashAttention inference-decode on Ascend NPU
type: kernel
tags: [incre_flash_attention, attention, flash_attention, online_softmax, flash_decode, kv_cache, paged_attention, cube_vector_pipe]
techniques: [fp32_accumulation, tail_mask]
hardware_features: [unified_buffer, l0c_buffer, l1_buffer, cube_vector_1_to_2]
architectures: [Ascend910B1, Ascend910B2, Ascend910_93, Ascend950PR, Ascend950DT, Ascend310P1, Ascend310P3, Ascend310P5, Ascend310P7]
kernel_types: [attention]
operators: [incre_flash_attention]
symptoms: [precision_mismatch, ub_overflow, performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/incre-flash-attention
related: [technique-flash-attention-tiling, technique-flash-attention-pipeline, pattern-flash-attention-tiling-key-dispatch, kernel-flash-attention-score]
sources: [ops-transformer-incre-flash-attention]
performance_claims: []
---

# IncreFlashAttention (推理增量解码)

CANN ops-transformer 中的 `IncreFlashAttention`（IFA）算子在推理场景下实现**增量解码** FlashAttention：query 的 S 轴固定为 1，key/value 为经过 KV Cache 累积的历史 state（每个 batch 的实际 S 长度可能不同，输入为 padding 后的固定长度）。支持 Atlas A2（arch32）/ Atlas A3 训练（ascend910_93）/ Ascend 950PR/950DT（arch35）/ Atlas 推理加速卡 310P（arch20）。招牌机制为 **FlashDecode**（split-KV + lse 加权规约）。本页归纳算子语义与计算流程，可复用的 tiling/流水/dispatch 知识分别见 [[technique-flash-attention-tiling]]、[[technique-flash-attention-pipeline]]、`artifacts/incre-flash-attention/tiling_dispatch_tables.md`。变体对照与训练前向 [[kernel-flash-attention-score]]、推理 prefill `prompt_flash_attention` 共享 QK→softmax→PV 骨架，差异在于增量解码、KV cache 管理、FlashDecode。

## 1. 输入输出

### 输入（按 `op_host/incre_flash_attention_def.cpp` L22-271 与 V4 aclnn L88-354）

| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| query | 必选 | 查询，**S 轴固定 1** | FP16/BF16 |
| key | 必选(tensorlist) | 键，KV Cache | FP16/BF16/INT8 |
| value | 必选(tensorlist) | 值，KV Cache | 同 key |
| pse_shift | 可选 | 位置编码 | FP16/BF16 |
| atten_mask | 可选 | 注意力掩码，1=不参与 0=参与 | BOOL/INT8/UINT8 |
| actual_seq_lengths | 可选 | per-batch KV 实际长度 | INT64 |
| dequant_scale1 | 可选 | BMM1 后反量化因子 | UINT64/FLOAT32 |
| quant_scale1 | 可选 | BMM2 前量化因子 | FLOAT32/BF16 |
| dequant_scale2 | 可选 | BMM2 后反量化因子 | UINT64/FLOAT32 |
| quant_scale2 | 可选 | 输出量化因子 | FLOAT32/BF16 |
| quant_offset2 | 可选 | 输出量化偏移 | FLOAT32/BF16 |
| antiquant_scale | 可选 | 反量化因子（per-tensor/channel） | FP16/BF16/FP32 |
| antiquant_offset | 可选 | 反量化偏移 | 同 antiquant_scale |
| block_table | 可选 | PageAttention block 映射表 | INT32 |
| kv_padding_size | 可选（V4） | KV 左 padding | INT64 |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| attention_out | 注意力输出（shape 同 query） | FP16/BF16/INT8 |

### 维度约定

- layout：BSH / BNSD / BSND（V4 doc L288；def.cpp `input_layout` 默认 "BSH"）
- B：A2/950 ≤ 65536；310P ≤ 256。N ≤ 256。D ≤ 512。query S = 1。
- GQA：`numHeads` 是 `numKeyValueHeads` 的倍数；`numKeyValueHeads=0` 表示与 query 相等。

### 属性（def.cpp L288-293）

- `num_heads`（必需）：query head 数
- `scale_value`（默认 1.0）：QK 缩放系数（1/√d）
- `input_layout`（默认 "BSH"）：BSH/BNSD/BSND
- `num_key_value_heads`（默认 0）：KV head 数（GQA）
- `block_size`（默认 0）：PageAttention 每 block 最大 token 数（≤512，FP16/BF16 16 对齐，INT8 32 对齐，推荐 128）
- `inner_precise`（默认 1）：0=高精度 / 1=高性能

## 2. 核心计算公式

```
attention_out = Softmax(scale * (Q·K^T), atten_mask) · V
```

IFA query S1=1，故 `Q·K^T` 为 1×S2 行向量。采用 FlashSoftmax 在线刷新（对 Skv 轴切分）替代显式 softmax，FP32 中间累加。atten_mask 中 true 的位置 select 成负极小值，softmax 后变 0 实现遮蔽（设计介绍 L20）。

## 3. 计算流程（四阶段 CV + FlashDecode）

| 阶段 | 单元 | arch35 入口 | 操作 |
|------|------|------------|------|
| Cube1 (Bmm1) | AIC | `service/mm1_processor.h` | Q·K^T -> attention_score（FP32 累加，S1=1） |
| Vector1 (Vec1) | AIV | `service/vec1_processor.h` | pse add -> ×scale -> atten_mask select -> FlashSoftmax(在线 max/sum) |
| Cube2 (Bmm2) | AIC | `service/mm2_processor.h` | P·V（P 为 softmax 结果） |
| Vector2 (Vec2) | AIV | `service/vec2_processor.h` | rescale(÷softmax_sum) -> 可选 quant -> 写回 attention_out |

### FlashSoftmax 在线刷新（设计介绍 L22-32）

- `exp[i] = e^(max[i-1] - max[i])`
- `MM[PV][i] = MM[PV][i-1]·exp[i] + MM_new[PV][i]`
- 最终 `attention_out = PV_final / softmax_sum`

### FlashDecode 规约（设计介绍 L73/119-138）

当 `BN2 < 0.4 × 总核数` 时，对 S2 轴外切 SplitKv 份分配到不同核，各核独立做 Attention，最后 lse 加权规约：

- `scale[i] = exp(lse[i]) / Sum(exp(lse[i]))`
- `res = Σ split_res[i] · scale[i]`

## 4. 模板与变体

### 4.1 arch32 模板（设计介绍 L40-57，IFA 实际调用 3 种）

| 模板 | 文件 | perfMode | 特点 |
|------|------|----------|------|
| C+V 基础模板 | `arch32/incre_flash_attention_split_Bbn2s2_Us2.h` | 0 (C1_V2) | Cube 做 matmul（高阶 API），顺序流水，绝大多数场景 |
| 全 Vector 模板 | `incre_flash_attention_allvec_new.h`（op_kernel/） | 1 | matmul 由 vector 实现，降 Cube 启动/CV 通信开销；推理加速卡全用，A2 非 PA/非 GQA/全 FP16 |
| matmul 基础 API 模板 | `arch32/incre_flash_attention_preload.h` | 3 | CUBE 视角 + N-Buffer 流水，CUBE 核内 Buffer 两 matmul 间统一调度 |

> `preload_dd`（伪量化 MSD DD，perfMode 6）与 `preload_mla`（MLA 全量化，perfMode 5）两种模板设计介绍 L54/L56 明确标注**仅 FIA 调用、IFA 不调用**。

### 4.2 版本对照

| 版本 | aclnn | 相对前一版本新增 |
|------|-------|------------------|
| V1 | `aclnnIncreFlashAttention` | 基础增量解码（FP16/INT8，PA/量化/pse） |
| V2 | `aclnnIncreFlashAttentionV2` | 扩展（BF16 等） |
| V3 | `aclnnIncreFlashAttentionV3` | 扩展 |
| V4 | `aclnnIncreFlashAttentionV4` | **KV 左 padding**（V4 doc L18） |

## 5. 架构与产品支持

| 产品 | 架构代号 | config（def.cpp） | kernel 入口 | tiling |
|------|----------|-------------------|-------------|--------|
| Atlas A2 训练/推理 | arch32 (910B) | `ascend910b` L303 | `incre_flash_attention` | `tiling.cpp` |
| Atlas A3 训练 | ascend910_93 | `ascend910_93` L304 | arch32 | `tiling.cpp` |
| Ascend 950PR/950DT | arch35 (DAV_3510) | `ascend950` L306-316 | `incre_flash_attention_apt` | `tiling_v2.cpp` |
| Atlas 推理加速卡 | arch20 (310P) | `ascend310p` L318-408 | `incre_flash_attention` | `tiling.cpp` 310P 分支 |

- 950 走 arch35 regbase 服务化（mm1/vec1/mm2/vec2 processor），`jitCompile static_false,dynamic_false`。
- 310P 仅 FP16（部分 INT8 KV），经 arch32 入口转 `arch20/*` 解码器（关系 `needs_verification`）。
- V4 doc L9 标注 950PR/950DT ×（V4 KV 左 padding 特性），与 def.cpp 注册 950 存在差异，V4 在 950 支持范围 `needs_verification`。
- Atlas 200I/500 A2、Atlas 训练系列（A1）不支持。

## 6. 特性机制

### 6.1 FlashDecode（split-KV + lse 规约）

IFA 招牌。核间先按 BN2 分核，BN2 不足 0.4×核数时对 S2 外切 SplitKv 份，总块数 `BN2 × SplitKv`，各核独立 Attention 后 lse 加权求和规约（设计介绍 L73/119-138）。可迁移为 Triton 两段式（split kernel 输出 partial lse + partial o；combine kernel 加权 reduce），见 `artifacts/incre-flash-attention/incre-flash-attention-triton-skeleton.py`。

### 6.2 AntiQuant-MSD（反量化优化）

INT8 KV 场景，经典反量化需对整个 KV 矩阵搬入 Vector 反量化（数据量大、性能差）。IFA 变换 Q 适配 KV：Q 展开成多行 int8 -> `AA·B`（int8×int8=int32）-> reduce（设计介绍 L140-155）。可迁移为 `tl.dot(int8, int8, out_dtype=tl.int32)` + reduce。

### 6.3 PageAttention

`block_table` 非空时启用，KV 按 block 索引在连续内存排布，input_layout 失效。CANN MatMul 提供回调函数做 B 矩阵 GM->L1 拷贝（Cube 执行，Vector 经 GM 通知，设计介绍 L156-158）。可迁移为 `tl.load(block_table)` 查物理偏移 -> 间接 `tl.load` KV。

### 6.4 GQA

G = queryHead/KvHead，Vector 上 G 轴切分由 UB 容量决定：`g = target_ub_size / column_size`（设计介绍 L160-173）。

### 6.5 KV 左 padding（V4）

`kv_padding_size` 非空时，KV 搬运起点 = `Smax - kvPaddingSize - actualSeqLengths`，终点 = `Smax - kvPaddingSize`，越界返回 0（V4 doc L255-263）。

## 7. Agent actions

1. 确认目标架构（arch32 / arch35 / arch20）与 dtype，选择对应 tiling 路径与模板。
2. 确认 query S1=1（增量解码）；按 layout 推导 B/N2/G/S2 轴。
3. 评估 `BN2 < 0.4 × 核数`：是则使能 FlashDecode（split-KV + lse 规约两段式），否则单 kernel 沿 S2 全长在线 FlashSoftmax。
4. 检查 `block_table` 是否非空 -> PagedAttention 间接 load；`antiquant_*` 非空 + INT8 KV -> AntiQuant-MSD；`kv_padding_size` 非空 -> V4 KV 左 padding 偏移。
5. 参考 [[technique-flash-attention-pipeline]] 安排 Cube/Vector 流水；IFA 三模板（C+V/全V/preload）按 dtype/layout/shape 选择。
6. Triton 实现参考 `artifacts/incre-flash-attention/incre-flash-attention-triton-skeleton.py`（pseudocode，含 FlashDecode 两段式）。
7. MLA 全量化与伪量化 MSD DD 模板 IFA 不调用（设计介绍 L54/L56），仅 FIA 使用，迁移时无需覆盖。

## 8. 复现性声明

本页结论基于源码静态分析（confidence: inferred），tiling 参数数值、流水 bound、arch35 模板枚举、V4 在 950 支持范围未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。FlashDecode lse 精确公式以 `vf_softmax_const.h` 为准（`needs_verification`）。来源见 [[ops-transformer-incre-flash-attention]]。
