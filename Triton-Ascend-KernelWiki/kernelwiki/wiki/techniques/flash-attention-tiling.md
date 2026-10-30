---
id: technique-flash-attention-tiling
title: FlashAttention tiling and core-partition strategy on Ascend NPU
type: technique
tags: [tiling, flash_attention, attention, core_partition, ub_tiling, cv_tiling_separation]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, l1_buffer, l0c_buffer, l2_cache]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR]
kernel_types: [attention]
operators: [flash_attention_score]
symptoms: [ub_overflow, performance_bottleneck, precision_mismatch]
confidence: inferred
reproducibility: snippet
sources: [ops-transformer-flash-attention-score]
prerequisites: [了解Ascend NPU架构、CUBE/VECTOR计算单元分工、FlashAttention基本算法]
related: [pattern-flash-attention-tiling-key-dispatch, technique-flash-attention-pipeline]
---

# FlashAttention tiling 与分核策略

FlashAttention 算子的 tiling 策略解决两个核心问题：**如何将数据分配到多核并行计算**（核间切分），**如何在单核有限的 UB/L1/L0C 内完成计算**（核内切分）。Ascend NPU 上 AIC（Cube）与 AIV（Vector）分离的架构进一步引入 CV tiling 分离策略。

## 1. CompileInfo（硬件平台参数）

Host 侧 `TilingPrepare` 从平台 API 读取：

| 参数 | 含义 |
|------|------|
| aivNum | Vector 核数 |
| aicNum | Cube 核数 |
| ubSize | Unified Buffer 大小（Byte） |
| l1Size | L1 Cache 大小（Byte） |
| l0cSize | L0C 大小（Byte） |
| l2CacheSize | L2 Cache 大小（Byte） |
| socVersion | SoC 版本 |
| npuArch | NPU 架构代号 |

## 2. arch32（A2 训练系列）多模板设计

FA 算子包含 B、N2、G、S1、S2 共 5 个轴，切分顺序：**先核内再核间**。核内切分依据基本块大小选择切分轴，核间切分把核内切分后剩余轴按 AI Core 核数分配。不同 shape 导致不同切分轴，因此拆分为 6 类模板：

| 模板 | 核内切分轴（UB） | 核间切分轴（Block） | 进入条件 |
|------|-------------------|---------------------|---------|
| TNDSameAB | S1 | B, N2, G | TND 且 (B≥4 且 S1≥8192 且 S2≥8192 且 maxS1≥512 且 maxS2≥512) 或 (maxS2≥5120 且 maxS1≥C5120) |
| TND | S1, S2 | B, N2, G, S1 | TND 但不满足 TNDSameAB |
| SameAB | S1 | B, N2, G, S1 | 非 TND，非 FP32，S2≥512 且 (D%16≠0 或 D=96)，或 S2>1024 且 128<D<196 |
| S1S2 | S1, S2 | B, N2, G, S1 | S2>1024，不满足上述条件 |
| S1 | S1, D | B, N2, G, S1 | 中等 shape，不满足上述条件（见源码详细条件） |
| B | — | B | 小 shape 兜底 |

**模板选择优先级**：序号越小优先级越高，TNDSameAB > TND > SameAB > S1S2 > S1 > B。

### 2.1 CV tiling 分离（A2）

AIC 与 AIV 算力差异大，使用不同基本块，CV tiling 分离策略减少通信次数：

| 单元 | 基本块（A2） | 说明 |
|------|-------------|------|
| Cube (AIC) | C-Tiling: (S1_c_i, D)×(D, S2_c_i) → (128, 1024) | mm 输出 128×1024，存 workspace |
| Vector (AIV) | V-Tiling: (S1_v_i, S2_v_i) → (8, 1024) | 从 workspace 拷入，Vec 计算，写回 GM |

**nRatio=8** 配比：Cube 一次发射 128×1024 数据，Vector 分 128/8=16 次处理。1:16 CV 数据量配比让 Cube 一次发射大块数据，避免小块通信开销。

伪代码：
```c++
// C侧 matmul
Bmm((S1_c_i,D) × (D,S2_c_i))  // 128*1024, 输出到 workspace
// V侧 Vector
for S1_c_i/S1_v_i = 128/8:
  copy_gm_to_ub(S1_v_i * S2_v_i)  // 从 workspace 拷入 bmm 结果
  Vector(S1_v_i, S2_v_i)          // FlashSoftmax + mask + scale
  copy_ub_to_gm(S1_v_i * S2_v_i)  // 写回 GM
```

### 2.2 CV 基本块选择

| 模板 | S1.i（CV基本块） | S2.i（CV基本块） | 说明 |
|------|-------------------|-------------------|------|
| S1S2 | 默认 64；若 B*N2*G*S1.o > Vector核数则 128 | 1024 | S2 切分产生 FlashSoftmax 刷新流程 |
| S1 | 依据 S2 动态调整，尽量让 S1*S2 大 | 不切分，=S2 | S2≤1024 无需刷新 |
| B | — | — | batch 轴切分，B.i*N2*G*S1*D 或 B.i*N2*G*S1*S2 需 < L1 Size |

## 3. arch35（950PR/950DT）统一模板

Ascend 950PR/950DT 核内切分基本块为 128×128，各 shape 性能均可达最优水平，因此**只设计一套模板**，不区分 layout 是否 TND。

| CV 基本块（arch35） | 说明 |
|------|------|
| S1.i: 128（少量场景 64） | — |
| S2.i: 128（少量场景 256） | — |

AIC&AIV 融合架构下 AIC 可直接输出到 UB，极大降低搬运时间。但 Dv>128 时 mm2 输出仍需先保存到 GM。

## 4. 核间切分策略

- **多核用满原则**：核心数量用满，防止部分核闲置。
- **负载均匀原则**：每个核心被分配的计算量相对均匀，避免某些核数据量过大而其他核空闲。
- **CV 算力匹配原则**：AIC 和 AIV 处理的数据量要符合对应算力，避免长时间空闲。
- **S2 不切到多核**：所有 FA 模板都不会把 S2 轴切分到多核（FlashSoftmax 需要连续的 S2 分块做在线更新）。

## 5. UB 内存分配

arch32 `AttentionScoreSingleCoreTensorSize` 关键字段：

| UB 区域 | 用途 |
|---------|------|
| bmm1ResUbSize / mmResUbSize | QK matmul 结果（FP32） |
| attenMaskUbSize | attention mask |
| pseUbSize | 位置编码 |
| dropMaskUbSize | dropout mask |
| softmaxMaxUbSize | FlashSoftmax 行 max（FP32） |
| softmaxSumUbSize | FlashSoftmax 行 sum（FP32） |
| softmaxExpUbSize | FlashSoftmax 行 exp（FP32） |
| bmm2ResUbSize | PV matmul 结果（FP32） |
| apiTmpBufferBytes | Softmax/Dropout API 临时缓冲区 |

**double buffer**：V 侧 ping-pong 两份独立内存，DataCopy 与 Clc 交替流水。

arch35 `TensorSizeParams` 类似，额外含 `inputQueBytes`/`outputQueBytes`（Cube 队列字节）及 workspace 分区偏移（`wkspSection1OffsetBytes`/`wkspSection2OffsetBytes`）。

## 6. Workspace

- 空tensor 场景：预留 **100MB** workspace。
- 正常场景：workspace 分为两个 section，分别存放 mm1 结果临时区域和 mm2/bmm2 相关临时区域（具体偏移由 tiling 计算）。

## 7. Agent actions

1. 确认目标架构后选择 tiling 路径：arch32 多模板 vs arch35 统一模板。
2. 按输入 shape 评估模板优先级：TNDSameAB > TND > SameAB > S1S2 > S1 > B。
3. 根据 S2 是否 >1024 决定是否需要 FlashSoftmax S2 切分刷新流程。
4. 检查 UB 容量是否满足 bmm1Res + mask + pse + softmax + bmm2Res + tmp ≤ ubSize。
5. 安排核间切分确保多核用满、负载均匀、CV 算力匹配。
6. S2 不切到多核，保证 FlashSoftmax 在线更新的正确性。

来源与详细分析见 [[ops-transformer-flash-attention-score]]。
