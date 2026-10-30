---
id: technique-flash-attention-pipeline
title: FlashAttention Cube-Vector pipeline orchestration on Ascend NPU
type: technique
tags: [pipeline, flash_attention, attention, cube_vector_pipe, double_buffer, preload]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, l0c_to_ub, cube_vector_1_to_2]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR]
kernel_types: [attention]
operators: [flash_attention_score]
symptoms: [ub_overflow, performance_bottleneck, precision_mismatch]
confidence: inferred
reproducibility: snippet
sources: [ops-transformer-flash-attention-score]
prerequisites: [了解FlashAttention四阶段计算流程(QK→softmax→PV→output)、CV流水基本概念]
related: [technique-flash-attention-tiling, pattern-flash-attention-tiling-key-dispatch]
---

# FlashAttention 流水编排

FlashAttention 融合了 Cube（matmul）与 Vector（softmax/elementwise）两类计算单元，需要精心编排流水使两者并行、减少空闲。不同架构有不同的 CV 协同方式。

## 1. 四阶段 Pipeline

| 阶段 | 计算单元 | 入口函数 | 操作 | 数据依赖 |
|------|----------|----------|------|---------|
| Stage 1: Bmm1 | Cube (AIC) | IterateBmm1 | Q·Kᵀ matmul | 依赖 Q、K 的 L1 加载 |
| Stage 2: Vec1 | Vector (AIV) | ProcessVec1 | pse add/mul → scale → mask select → FlashSoftmax (在线 max/sum/exp) | 依赖 Bmm1 输出 (workspace/UB) |
| Stage 3: Bmm2 | Cube (AIC) | IterateBmm2 | P·V matmul | 依赖 Vec1 的 softmax 结果 (P 矩阵) + V 的 L1 加载 |
| Stage 4: Vec2 | Vector (AIV) | ProcessVec2 | rescale (÷ softmax_sum) → dropout → output 写回 GM | 依赖 Bmm2 输出 + softmax_max/sum |

**关键依赖链**：Bmm1 → Vec1 → Bmm2 → Vec2。Bmm1 与 Bmm2 是 Cube 计算，Vec1 与 Vec2 是 Vector 计算。Cube 与 Vector 之间通过 workspace（A2）或 UB（950PR）传递数据。

## 2. V 侧流水（double buffer）

V 侧使用 **ping-pong double buffer** 实现数据搬运与计算的流水重叠：

```
时间轴:  |----|----|----|----|----|----|----|
ping:   |DataCopy|  Clc  |      |DataCopy|  Clc  |
pong:   |      |DataCopy|  Clc  |      |DataCopy|  Clc  |
```

- ping 和 pong 两块计算数据占用独立内存资源。
- 同一片数据：DataCopy → Clc 串行；不同数据切片：并行处理。
- 不同 shape 的 FlashSoftmax 过程较复杂，简单 double buffer 无法覆盖所有情况，因此存在不同的计算流水排布。

## 3. CV 流水 — Atlas A2 训练系列（arch32）

### 3.1 Cube 双发（preload）

A2 上 C:V=1:2（一个 Cube 核对应两个 Vector 核）。Vector 计算节点少、速度快，Cube 搬运时长足以覆盖 Vector 计算时长。

**Cube 双发机制（CV 间 preload 流水）**：
1. 提前发射**两块** Cube 计算（Cube1、Cube2）。
2. Cube1、Cube2 计算可以衔接，Cube 利用率最高。
3. 只要关注 Cube 的 MTE2 耗时，最终达成 **MTE2 bound** 或 **Cube bound**。

### 3.2 CV 数据通路

A2 架构下 AIC 与 AIV 之间数据交互通路为 **L2** 和 **GM**（Global Memory）。两者之间的交互次数对性能影响较大，CV tiling 分离策略旨在减少通信次数。

### 3.3 L1 复用（EnableL1Reuse）

部分模板支持 `EnableL1Reuse`：Bmm2 的左矩阵（softmax 结果 P）可留在 L1 供多次 Bmm2 使用，减少从 GM 的重复加载。当 `Bmm2Source=L1` 时启用。

## 4. CV 流水 — Ascend 950PR/950DT（arch35）

### 4.1 AIC&AIV 融合架构

950PR 同时支持 AIC&AIV 分离架构和 AIC&AIV 融合架构。融合架构下：

- AIC 可**直接输出到 UB**（Unified Buffer），AIV 利用 UB 上的数据做 Vector 运算。
- UB 是 AIC&AIV 之间比 GM 更高效的交互通路。
- mm1 的输出**始终保存在 UB**（不需要写到 workspace）。
- Dv>128 时 mm2 的输出**依旧需要先保存到 GM**（UB 容量不够）。

### 4.2 Cube 三次 preload

950PR 的 Cube preload 次数为 **3 次**：完成 3 次 mm1 计算后才会开启 mm2 计算。

目的：优化启动阶段的 CV 流水，使其更紧密，达到整体性能最优。

### 4.3 TSCM（L1 reuse）

arch35 使用 `FaTscm`（TSCM buffer）管理 L1 复用。`MatmulPolicyType` 控制：
- `NORMAL`: 标准模式
- `UNSPLITK`: 不做 SplitK，L1 可完全复用

## 5. Matmul 格式选择

| 格式 | 说明 | 使用场景 |
|------|------|---------|
| Bmm1Format=ND | mm1 左矩阵 Q 使用 ND 格式 | 默认 |
| Bmm1Format=NZ | mm1 左矩阵 Q 使用 NZ 格式 | SameAB 模板中 Cube 为主核时，L0A 可复用左矩阵 |
| Bmm2Source=GM | mm2 左矩阵 P 从 GM 读取 | 默认 |
| Bmm2Source=L1(TSCM) | mm2 左矩阵 P 从 L1(TSCM) 读取 | S1 模板中 P 可留在 L1 复用 |

NZ 格式在 SameAB（Cube 为主核）模板中有优势：V0、V1 的 Matmul 任务可复用左矩阵，且输出的部分结果可在 L0C 累加，减少对带宽的依赖。

## 6. 编程视角

### 6.1 AscendC 高阶 API

两种编程模式：

1. **Vector 为主核，Cube 为从核**（默认）：Vector0 和 Vector1 独立发起 Matmul 任务，两者无关联。kernel 模板名不以 `_sab` 结尾。

2. **Cube 为主核，Vector 为从核**（SameAB 模板，`_sab` 后缀）：V0 统一发起 Matmul 任务，V0 和 V1 各处理一半。V0/V1 的 Matmul 可复用左矩阵，L0C 累加部分结果。大部分场景性能更优。

### 6.2 AscendC 低阶 API

arch35 (950PR) 上 FA 通过 AscendC 低阶 API 实现，更加彻底地使用 Cube 为主核、Vector 为从核，Matmul 任务完全从 Cube 侧发起，通过同步通知 Vector 侧。

## 7. Agent actions

1. 确认架构后选择 CV 流水策略：A2 双发 vs 950PR 三次 preload。
2. 安排 V 侧 double buffer：ping-pong 两份 UB 空间。
3. 评估是否可启用 L1 复用（EnableL1Reuse / Bmm2Source=L1）。
4. SameAB 场景考虑 Cube 为主核 + NZ 格式，利用 L0C 累加优势。
5. 检查 UB 容量：mm1 结果 + softmax 临时 + mm2 结果 + API 临时 ≤ ubSize。
6. Dv>128（950PR）时 mm2 输出走 GM 通路，需额外 workspace。

来源与详细分析见 [[ops-transformer-flash-attention-score]]。
