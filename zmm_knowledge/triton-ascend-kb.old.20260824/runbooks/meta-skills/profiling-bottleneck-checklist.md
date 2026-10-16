---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: "6.1"
title: Profiling 驱动的瓶颈诊断与优化路由
description: 用 msprof 和 simulator 识别 scalar fallback、内存瓶颈、流水线 stall、Cube 利用率低四类问题，按指标路由到对应 runbook。
tags: [profiling, msprof, bottleneck, aiv_scalar_ratio, aiv_mte2_time, pipe_utilization]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## DG-12 Profiling 驱动的瓶颈诊断与优化路由

【优化维度】：Agent 元技能｜瓶颈诊断树
【泛化层级】：Triton通用（迁移条件：所有 Triton-Ascend kernel 性能调优；无失效边界）

<原则：Ascend 优化必须先看 profiling 数据再动手。`op_summary_*.csv` 中的 `aiv_scalar_ratio`、`aiv_mte2_time`、`aiv_vec_ratio`、`PipeUtilization` 四个指标可直接定位 80% 的瓶颈。按指标路由到对应 runbook，避免盲目调参。>

**核心指标与路由表**：

| 指标 | 症状 | 诊断 | 修复方向 |
|---|---|---|---|
| `aiv_scalar_ratio` > 10% | Vector 时间低，标量单元繁忙 | int64/int32 比较/加法降级 | [整数比较转 float32](../compute-pipeline/int-compare-cast-float32.md) |
| `aiv_mte2_time` > 2x 理论 | Load 延迟高 | bank conflict / 不对齐 / tile 太小 | [对角 core mapping](../compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md)、[512B 对齐](../memory-sram/row-width-512b-align.md)、[tiling 预算](../memory-sram/tiling-budget-and-layout-driven-selection.md) |
| `PipeUtilization` 低 | 流水线阶段不重叠 | load/compute/store 串行 | [care_padding=False](../compute-pipeline/care-padding-false-ilp.md)、[for-loop tiling](../compute-pipeline/for-loop-tiling-ilp.md)、[async_task](../compute-pipeline/async-task-cube-vector.md) |
| `aiv_vec_ratio` < 30% | elementwise Vector 利用率低 | BLOCK_SIZE 小 / 离散访问 / scalar | 增大 BLOCK、连续化索引、检查 int 降级 |
| `aiv_cube_time` 低 | matmul Cube 利用率低 | BLOCK 过小 / 非 256 倍数 | [tiling 预算](../memory-sram/tiling-budget-and-layout-driven-selection.md) |

**msprof 命令**：

```bash
msprof op --kernel-name=target_kernel --output=$HOME/output python3 test.py
```

**关键输出文件**：

| 文件 | 内容 |
|---|---|
| `PipeUtilization.csv` | MTE2/Vector/Cube/MTE3 各阶段利用率 |
| `op_summary_*.csv` | `aiv_vec_ratio`、`aiv_scalar_ratio`、`aiv_mte2_time` 等 |

**理论性能基线**：

```
Memory_time = Data_volume / 1.8e12
Compute_time = FLOPs / peak_TFLOPS
```

若 `measured / theoretical > 2x`，说明存在明显瓶颈，需按上表路由。

**完整诊断流程**：

```
1. 跑 baseline（无 profiling）记录总时间
2. 计算 theoretical time
3. 若 ratio < 0.5，用 msprof 收集指标
4. 优先看 aiv_scalar_ratio（修复最便宜）
5. 再看 aiv_mte2_time（bank conflict / 对齐）
6. 再看 PipeUtilization（流水线 stall）
7. 必要时用 simulator 看指令级 trace
8. 应用对应 runbook 的修复
9. re-profile 验证 ratio > 0.8
```

**坏实践（反例）**：
```python
# 不看 profiling，直接增大 BLOCK
BLOCK_SIZE = 4096  # 可能 UB 超限或 tile 过大降低并行度
```
后果：可能反向优化。✅ 正解即先 profile 再按指标路由。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_perf_003` — 内存访问优化 （收益 30%）
  - 适用于任意维度张量的元素级运算，核心是“一次连续化+一维遍历”，torch.empty_like创建的输出默认连续，无需额外处理；输入输出同形状算子无需reshape
- `exp_perf_008` — 内存初始化优化 （收益 8%）
  - 适用于所有Triton算子的输出张量创建场景，`torch.empty_like()`相比`torch.zeros_like`可节省约8%的初始化时间，且默认连续的特性适配一维访问优化，910b2/910b3平台小张量（<1e5元素）的初始化开销占比更高
- `exp_perf_009` — 冗余操作优化 （收益 10%）
  - 适用于所有输入输出同形状的element-wise算子（如relu、silu、加减乘除等），避免冗余reshape可节省10%左右的开销，尤其对高维张量（4D/5D）效果更明显
- `exp_perf_010` — 算子拆分优化 （收益 25%）
  - 适用于包含多阶段计算的复杂算子（如卷积+BN+激活、Attention全流程等），拆分后单个kernel编译时间缩短40%，910b2/910b3平台的调度并行度提升25%以上
- `exp_perf_012` — 标量消除 （收益 35%）
  - 此优化对由tl.arange生成的索引比较尤其有效。当参与比较的变量本身就是tl.full构造的fp32标量时，可直接使用而无需额外转换。注意fp32比较对mask逻辑无精度损失。
- `exp_perf_013` — 标量消除 （收益 15%）
  - 当min()涉及编译期常量时（如两个tl.constexpr），编译器可自动求值，无需替换。此优化主要针对至少一个操作数为运行时值的情况。
- `exp_perf_014` — 标量消除 （收益 10%）
  - 当所有相关维度都是constexpr时，优化效果最大；若涉及运行时值，需评估替代方案引入的额外tl.where开销是否值得。
- `exp_perf_015` — 内存访问优化 （收益 30%）
  - 此方法要求原始张量布局允许按期望方向连续读取。如果原始数据布局本身就是列优先，可能需要预先在host端转置或在kernel内使用其他数据编排策略。

<!-- okf:related:start -->

# 相关

- 相关主题: [Triton-Ascend kernel 调试排查清单](debugging-checklist.md) — 同属优化知识，主题相关
- 相关主题: [GPU-source 迁移到 Ascend NPU 的 memory/index 改造检查清单](gpu-to-ascend-migration-checklist.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
