---
name: triton-knowledge-autotuning-ingest
description: "分类5 调优空间设计（Autotuning Design）知识生产 skill：蒸馏候选 BLOCK_SIZE 剪枝、num_stages 搜索策略、autotune key 关联设计知识，产出 runbooks/autotuning/ 优化点。手动触发。"
disable-model-invocation: true
argument-hint: <轨迹路径或文档路径> [主题]
---

# triton-knowledge-autotuning-ingest — 分类5「调优空间设计」知识生产

覆盖二级维度 **5.1**，落点 `runbooks/autotuning/` + `ops/`。

## 提取点清单（本分类只收这些主题）

| 维度 | 主题 | 提取点 |
|---|---|---|
| 5.1 Autotuning Search Space Design | 搜索空间设计 | 候选 `BLOCK_SIZE` 剪枝、`num_stages` 搜索策略、autotune key 关联设计（**避免盲目暴力搜索导致编译耗时爆炸**） |

## 输入

同编排器 §2。轨迹重点看：autotune 配置（候选集、key）、调参迭代记录、编译耗时问题、最终选定配置与 baseline 对比。

## 工作流（五阶段）

### 阶段 1：采集

轨迹重点看：`@triton.autotune` 的 configs 列表、key 选择、prune 策略、调参前后性能数据、编译时间。

### 阶段 2：提取「诊断→修改→验证」三元组

- **诊断**：编译耗时爆炸 / 候选集无效（最优解总在边界）/ key 设计不当导致重复调参；
- **修改**：候选集剪枝（按 UB 约束/对齐约束预过滤）、key 重设计、stages 范围收敛；
- **验证**：调参时间变化 + 最终性能是否保住，须有记录。

### 阶段 3：蒸馏落盘

- 三元组 → `OPT-*` / `AP-*` / `CT-*`（autotune 与动态 shape 的交互坑多落 CT）。
- 条目按 [`STRUCTURE-runbook.md`](STRUCTURE-runbook.md) 骨架：占位名（`blockCandidates`/`numStages`/`tuneKey`…）、`dimension: 5.1`、泛化层级、「坏实践」必填。
- **剪枝规则必须是可执行的判据**（如"BLOCK_K × dtypeBytes 须整除 32B"），不收"适当减少候选"这类废话。

### 阶段 4：增量合并

同分类条目已存在 → 只 append「已知实例」；新点 → 新增 ID。**干净上下文子 Agent 合并**。

### 阶段 5：校验 + 图谱

自检 → `okf_graph.py candidates → judge → inject → viz → verify`。

## 关键原则

- **搜索空间设计是"减法知识"**：重点沉淀"哪些候选可以被证明无效"的判据，而不是"我调出了什么好参数"（那是算子案例卡的内容）。
- **key 设计关联泛化**：autotune key 的选择须与 shape 泛化策略一起记录（哪些维度进 key、哪些不进、为什么）。
- **编译耗时也是指标**：条目须同时记录调参时间收益与最终 kernel 性能，防止"省了编译、输了运行"。
