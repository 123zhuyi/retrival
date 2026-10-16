---
name: triton-knowledge-numerics-ingest
description: "分类3 算法与数值（Algorithmic & Numerics）知识生产 skill：蒸馏算法重构（Tree Reduction/Online Softmax/FlashAttention 分块）、混合精度、数值稳定、跨 Block 归约知识，产出 runbooks/algorithm-numerics/ 优化点与 ops/ 案例卡。手动触发。"
disable-model-invocation: true
argument-hint: <轨迹路径或文档路径> [算子名]
---

# triton-knowledge-numerics-ingest — 分类3「算法与数值」知识生产

覆盖二级维度 **3.1–3.4**，落点 `runbooks/algorithm-numerics/` + `ops/` + `reference/`。

## 提取点清单（本分类只收这些主题）

| 维度 | 主题 | 提取点 |
|---|---|---|
| 3.1 Algorithmic Complexity Optimization | 算法重构 | Tree Reduction 树形归约、Online Softmax、FlashAttention 分块重构（降低复杂度，适合片上单 Pass） |
| 3.2 Precision Strategy & Mixed Precision | 混合精度 | FP32 Accumulator 提升、BF16/FP8 选择策略、Cast 与 Compute 算子融合 |
| 3.3 Numerical Stability | 数值稳定 | Exponent Clamp、防 Overflow/Underflow、Log-Sum-Exp 变换（低精度 FP16/BF16 下尤其关键） |
| 3.4 Cross-Block Reduction & Communication | 跨 Block 归约 | Two-pass Reduction vs Online Reduction、Atomic 边界控制、Workspace Tensor 复用（Ascend 无 CUDA `__shfl_sync` warp 同步，须用上述机制弥补） |

## 输入

同编排器 §2。轨迹重点看：精度校验失败记录（MERE/MARE 不过）、cast 路径改动、归约算法改动、数值溢出相关修复。

## 工作流（五阶段）

### 阶段 1：采集

轨迹重点看：精度失败迭代的报错与数值差异、dtype 转换链、归约/softmax/attention 类算法结构、atomic 使用。

### 阶段 2：提取「诊断→修改→验证」三元组

- **诊断**：精度超标（含具体误差指标）、溢出/下溢、归约结果不确定（atomic 序）；
- **修改**：升累加精度、clamp、算法重构（online/tree）、归约两阶段化；
- **验证**：精度通过记录 + 性能变化（精度提升常伴性能代价，须同时记录两者）。

### 阶段 3：蒸馏落盘

- 三元组 → `OPT-*` / `AP-*` / `CT-*`（dtype 支持矩阵、数值边界类多落 CT）。
- 条目按 [`STRUCTURE-runbook.md`](STRUCTURE-runbook.md) 骨架：占位名（`accDtype`/`reduceAxis`/`maxVal`/`logSumExp`…）、`dimension`（3.1–3.4）、泛化层级、「坏实践」必填。
- **精度与性能必须成对记录**：只写精度收益不提代价的条目不收。
- 算子特定事实 → `ops/` 案例卡，双向回链。

### 阶段 4：增量合并

同分类条目已存在 → 只 append「已知实例」；新点 → 新增 ID。**干净上下文子 Agent 合并**。

### 阶段 5：校验 + 图谱

自检 → `okf_graph.py candidates → judge → inject → viz → verify`。

## 关键原则

- **数值结论必须有误差指标**：精度类条目引用具体校验口径（MERE/MARE/ULP 等）与通过/失败数据，不写"精度更好了"。
- **算法重构写清复杂度账**：Tree/Online 重构条目给重构前后的 pass 数与中间物化量对比。
- **跨 Block 归约强调 Ascend 特殊性**：无 warp shuffle，条目中禁止出现 CUDA warp 同步措辞，一律写成 two-pass/online/atomic 语境。
