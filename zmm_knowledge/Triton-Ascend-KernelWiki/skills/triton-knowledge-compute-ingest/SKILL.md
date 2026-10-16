---
name: triton-knowledge-compute-ingest
description: "分类2 计算与流水（Compute Engine & Execution）知识生产 skill：蒸馏 Vector/Cube 单元利用、Scalar 削减、流水掩盖、多核负载均衡、算子融合知识，产出 runbooks/compute-pipeline/ 优化点与 ops/ 案例卡。手动触发。"
disable-model-invocation: true
argument-hint: <轨迹路径或文档路径> [算子名]
---

# triton-knowledge-compute-ingest — 分类2「计算与流水」知识生产

覆盖二级维度 **2.1–2.4**，落点 `runbooks/compute-pipeline/` + `ops/` + `reference/`。

## 提取点清单（本分类只收这些主题）

| 维度 | 主题 | 提取点 |
|---|---|---|
| 2.1 AICore Compute Unit Utilization | 单元利用 | Vector/Cube 任务分工、Mask/Repeat 优化、Broadcast 消除、**Scalar 标量开销削减与高延迟 div/mod 索引化简**（Scalar Unit 算力弱，多维索引转换是隐形瓶颈） |
| 2.2 Compute & DMA Overlap | 流水掩盖 | Double/Triple Buffering 实现、Async Copy 异步搬运、**细粒度 Event Sync 替代粗粒度 PipeBarrier** |
| 2.3 Multi-Core Parallelism & Load Balancing | 多核负载 | Grid 设计（1D/2D/3D）、`program_id` 到物理 Core 的映射、Tail Core 空闲均分策略、**coreDim ≤ 65535 限制规避**（20~48 个 AICore 的调度与并发限制） |
| 2.4 Fusion Intelligence | 融合模式 | Element-wise / Reduction / Transpose 融合进 Reshape/Matmul 的 Prologue/Epilogue 融合，减少中间 Tensor 写回 GM 的 Round-trip |

## 输入

同编排器 §2：轨迹（`<轨迹集占位>`，重点看 grid/num_warps 等价物、流水线相关修改、多核切分改动）与文档（`<上游文档仓占位>`）。

## 工作流（五阶段）

### 阶段 1：采集

轨迹重点看：grid 设计与改动、div/mod 索引表达式、同步原语（barrier/event）、buffer 深度（num_stages 类）、融合相关的 kernel 合并动作。

### 阶段 2：提取「诊断→修改→验证」三元组

- **诊断**：Scalar 占比高 / DMA 空等 / Tail Core 空闲 / 中间 tensor round-trip，带 profiling 或报错出处；
- **修改**：索引化简、双缓冲、grid 重划、融合重构，带代码片段出处；
- **验证**：修改前后可复现数据，**必须指回轨迹 benchmark 记录**。

### 阶段 3：蒸馏落盘

- 三元组 → `OPT-*` / `AP-*` / `CT-*`（coreDim 上限、同步语义误用类多落 CT）。
- 条目按 [`STRUCTURE-runbook.md`](STRUCTURE-runbook.md) 骨架：占位名（`gridX`/`numCores`/`stageDepth`/`idxExpr`…）、`dimension`（2.1–2.4）、泛化层级、「坏实践」必填。
- 算子特定事实 → `ops/` 案例卡，双向回链。

### 阶段 4：增量合并

同分类条目已存在 → 只 append「已知实例」；新点 → 新增 ID。**干净上下文子 Agent 合并**。

### 阶段 5：校验 + 图谱

自检（frontmatter/`dimension`/锚点/占位名泄漏/数据出处）→ `okf_graph.py candidates → judge → inject → viz → verify`。

## 关键原则

- **Scalar 开销是 Ascend 特色知识**：div/mod 化简条目必须给出化简前后的索引表达式对照与出处。
- **负载均衡要给策略不给口号**：Tail Core 均分条目须含可代入的分配公式（余数如何摊到前 N 核）。
- **同步粒度是权衡**：Event Sync 替代 PipeBarrier 须写清适用条件（依赖关系可静态确定时）与失效边界。
- **融合判据**：融合条目须写清"为什么这里融合划算"（省掉的 GM round-trip 量级），不泛泛而谈。
