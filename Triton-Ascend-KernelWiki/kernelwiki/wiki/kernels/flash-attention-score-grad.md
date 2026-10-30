---
id: kernel-flash-attention-score-grad
title: FlashAttentionScoreGrad training-backward on Ascend NPU
type: kernel
tags: [flash_attention, attention, softmax, online_softmax, backward, gradient, matmul, cube_vector_pipe, recompute]
techniques: [fp32_accumulation, tail_mask]
hardware_features: [unified_buffer, l0c_to_ub, cube_vector_1_to_2, l1_buffer, l0c_buffer]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR]
kernel_types: [attention]
operators: [flash_attention_score_grad]
symptoms: [precision_mismatch, ub_overflow, performance_bottleneck]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/flash-attention-score-grad
related: [kernel-flash-attention-score, technique-flash-attention-tiling, technique-flash-attention-pipeline, technique-flash-attention-backward-recompute, pattern-flash-attention-tiling-key-dispatch]
sources: [ops-transformer-flash-attention-score-grad]
performance_claims: []
---

# FlashAttentionScoreGrad (训练反向)

CANN ops-transformer 中的 `FlashAttentionScoreGrad` 算子（简称 FAG）在训练场景下计算 `FlashAttentionScore` 的反向梯度，支持 Atlas A2 训练系列（arch32 / ascend910b / ascend910_93）与 Ascend 950PR/950DT（arch35）。本页归纳反向算子语义与计算流程；可复用的 tiling/流水/dispatch 知识见 [[technique-flash-attention-tiling]]、[[technique-flash-attention-pipeline]]、[[pattern-flash-attention-tiling-key-dispatch]]，反向重算方法见 [[technique-flash-attention-backward-recompute]]。前向算子见 [[kernel-flash-attention-score]]。

## 1. 输入输出

### 输入（按 op_host/flash_attention_score_grad_def.cpp 顺序）

| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| query | 必选 | 查询 Q | FP16/BF16/FLOAT/HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN |
| key | 必选 | 键 K | 同 query |
| value | 必选 | 值 V | 同 query |
| dy | 必选 | 输出梯度 dY | FP16/BF16/FLOAT（FP8 路径下同 qkv） |
| pse_shift | 可选 | 位置编码 | FP16/BF16/FLOAT |
| drop_mask | 可选 | dropout 掩码，1=保留 0=丢弃 | UINT8 |
| padding_mask | 可选 | padding 掩码 | FP16/BF16/FLOAT |
| atten_mask | 可选 | 注意力掩码，1=不参与 0=参与 | BOOL/UINT8 |
| softmax_max | 可选 | 前向 softmax 行最大值，加速重算 | FLOAT |
| softmax_sum | 可选 | 前向 softmax 行求和，加速重算 | FLOAT |
| softmax_in | 可选 | 前向 softmax 后的 P | FP16/BF16/FLOAT |
| attention_in | 可选 | 前向输出 Y=PV，参与 FlashSoftmaxGrad | FP16/BF16/FLOAT |
| prefix / actual_seq_qlen / actual_seq_kvlen / q_start_idx / kv_start_idx | 可选 | 变长/偏移（TND、varlen） | INT64 |
| d_scale_q/k/v/dy/o | 可选 | FP8 全量化反量化参数 | FLOAT |
| query_rope / key_rope | 可选 | RoPE 编码张量 | 同 query |
| sink / ds_scale / p_scale | 可选 | attention sink / P 缩放（V3+） | FLOAT |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| dq / dk / dv | Q/K/V 的梯度 | FP16/BF16/FLOAT |
| dpse | pse 的梯度 | FP16/BF16/FLOAT |
| dq_rope / dk_rope | RoPE 梯度（V3+） | FP16/BF16/FLOAT |
| dsink | sink 梯度（V3+） | FLOAT |

### 维度约定（README L187-194）

以 BNSD/BSND 为例（BSH/SBH 下 H=N*D）：B: 1~2M（prefix 时最大 2K）；N: 1~256；S: 1~1M；D: A2/A3 训练 1~512，950PR 1~768。内核计算逻辑轴：B、N2（kv head）、G（query/kv head 比）、S1（query seq）、S2（kv seq）、D、Dv。

### 属性（def.cpp L222-260）

`scale_value`(1.0)、`keep_prob`(1.0)、`pre_tockens`(INT_MAX)、`next_tockens`(INT_MAX)、`head_num`(必需)、`input_layout`(必需 BSH/SBH/BNSD/BSND/TND)、`inner_precise`(0)、`sparse_mode`(0，0=NO_MASK..8=BAND_LEFT_UP_CASUAL)、`pse_type`(1)、`seed`(2)、`offset`(0)、`out_dtype`(0)、`softmax_in_layout`("")。

## 2. 核心反向公式（6 步，FAG算子设计介绍.md L9-14）

设 `S=Mask(QK^T/√d + pse, atten_mask)`，`P=Dropout(Softmax(S),keep_prob)`，`Y=PV`：

| 步 | 公式 | 说明 |
|----|------|------|
| 1 重算 p | `p = SimpleSoftmax(Mask(Matmul(query, key^T) + pse) * scale)` | 重算前向 softmax 结果 P，存 UB |
| 2 dp | `dp = Dropout(Matmul(dy, value^T))` | dY·V^T |
| 3 ds | `ds = p * Sub(dp, FlashSoftmaxGrad(dy, attention_in))` | FlashSoftmaxGrad 入参 dY 与前向输出 Y |
| 4 dq | `dq = Matmul(ds, key) * scale` | ds·K |
| 5 dk | `dk = Matmul(ds^T, query) * scale` | ds^T·Q |
| 6 dv | `dv = Matmul(Dropout(p)^T, dy)` | Dropout(P)^T·dY |

FlashSoftmaxGrad 实质：`ds = P ⊙ (dP − rowsum(dP ⊙ P))`，`dP = dY·V^T`（步 2）。反向**重算** P（softmax_max/sum 作为可选输入可加速重算，见 def.cpp L72-83）。共 4 次 matmul：mm1=Q·K^T（重算）、mm2=dy·V^T、mm3=ds·K / ds^T·Q、mm4=P^T·dy。

## 3. 计算流程（3 阶段 Pipeline，FAG算子设计介绍.md L20-23）

| 阶段 | 入口函数 | 单元 | 对应步 |
|------|----------|------|--------|
| Cube1 | ComputeMM1 | AIC | 步 1（重算 p：QK^T + softmax） |
| Vector1 | ComputeVec | AIV | 步 2-3（dp、ds：FlashSoftmaxGrad + elementwise） |
| Cube2 | ComputeMMDqkv | AIC | 步 4-6（dq/dk/dv matmul） |

V 侧 double buffer（ping-pong）；CV 间 preload（Cube 提前发射）。arch32 C:V=1:2 核数比，1:16 数据配比（nRatio=8）；arch35 C:V=1:2，L0C->UB / UB->L1 高速通路。详见 [[technique-flash-attention-pipeline]]。

## 4. Tiling 与多模板

- **arch32**：8 类模板，按 shape 路由（FAG算子设计介绍.md L264-319）：S1S2（通用泛化）、SameAB（_sab，Cube 为主核，确定性/大序列）、BN2（S1<1024&&S2<1024&&(B*N2*2>CoreNum‖G==1)）、NGS1S2_BN（G*S1*S2≤32KB&&G==1&&S2<1536）、B（N2*G*S1*S2≤64*128）、Basic/MLA、Basic_det（确定性）、Empty。CV 基本块 C 128×1024 / V 8×1024。
- **arch35**：4 类模板（BN2: S1≤128&&S2≤128&&G==1；BN2S2: TND&&B*N2*S2>cubeCoreNum&&G==1；BN2GS1S2 通用；确定性），仅 2 个 kernel 文件（非确定性/确定性），模板参数隔离。CV 基本块 C 128×128 / V 64×128。
- 切分顺序：先核内再核间；S2 不切多核（softmax 在线更新需连续 S2）。详见 [[technique-flash-attention-tiling]]。

## 5. 模板 dispatch（op_kernel/flash_attention_score_grad.cpp L488-901）

20 个 kernel 模板参数，按 `UB0/UB1/Block/IsSameAB/IsDeterministic` 分发：空 tensor(0,0,0) / SameAB(4,3,4,1 → key 0x1434) / S1S2(4,3,4,0) / BN2(4,3,1 → 0x134) / NGS1S2_BN(9,9,1 → 0x199) / B(9,9,0 → 0x099) / Basic_det(9,9,9,Det=1 → 0x1999) / Basic(9,9,9 → 0x999)。按 dtype（FP16/BF16/FLOAT）特化，FP32 仅 S1S2/BN2/BN/B。dispatch 维度（dtype/layout/has_mask/has_pse/has_dropout/has_rope/deterministic/sparse/NZ 输出/L1 复用/对齐特化）可映射为 Triton `tl.constexpr` 分支与 `BLOCK_*` 选择。详见 artifact `tiling-dispatch-tables.md` 与 [[pattern-flash-attention-tiling-key-dispatch]]。

## 6. 架构与产品支持

| 产品 | 架构代号 | 支持 | 说明 |
|------|----------|------|------|
| Atlas A2 训练 | arch32 (ascend910b) | √ | AIC&AIV 分离，多模板（8 类），aclnn V1-V3 |
| Atlas A3 训练 | ascend910_93 | √ | 复用 arch32 kernel |
| Ascend 950PR/950DT | arch35 (DAV_3510) | √ | AIC&AIV 融合，统一模板，aclnn V4 |
| Atlas A2/A3 推理 | - | × | 仅训练场景 |

FP8（FLOAT8_E5M2/E4M3FN/HIFLOAT8）仅 arch35(950) 支持，且不可配 pse/dropMask/attenMask/keepProb（README L179-181）。aclnn 版本：V1-V3 仅 A2 训练（950PR ×），V4 仅 950PR（A2 ×）；V2 加 pseType，V3 加 sink/dsink，V4 合并 V2+UnpaddingV2+varlen+自动 Dropout（docs/aclnnFlashAttentionScoreGrad*.md）。

## 7. Agent actions

1. 确认架构（arch32 vs arch35）与 dtype，选择对应模板分发路径与 aclnn 版本（V1-V3 A2 / V4 950）。
2. 反向需重算 P：按 [[technique-flash-attention-backward-recompute]] 安排 6 步计算与 4 次 matmul。
3. 按输入 layout 与 shape 推导核内/核间切分轴，参考 [[technique-flash-attention-tiling]]；S2 不切多核。
4. 检查 atten_mask/dropMask/pse/softmax_max/sum/attention_in 是否存在，选对应 tiling key bit / constexpr 分支。
5. 参考 [[technique-flash-attention-pipeline]] 安排 Cube/Vector 流水与 double buffer；arch35 利用 L0C->UB/UB->L1。
6. Triton 实现参考 `artifacts/flash-attention-score-grad/flash-attention-score-grad-triton-skeleton.py`（pseudocode）。
7. 空 tensor 走短路清零（FAG_EMPTY_TILING_KEY，100MB workspace）。

## 8. 复现性声明

本页结论基于源码静态分析（confidence: inferred），tiling 参数数值、流水 bound、FP8/BF16 精度边界、aclnn 运行时契约未经硬件实测。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 [[ops-transformer-flash-attention-score-grad]]。
