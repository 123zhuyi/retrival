---
name: triton-knowledge-memory-ingest
description: "分类1 内存与切块（Memory & SRAM）知识生产 skill：从 Agent 轨迹/官方文档蒸馏 Triton-Ascend 访存效率、Tiling/SRAM 预算、ND↔NZ、32B/512B 对齐知识，产出 runbooks/memory-sram/ 优化点与 ops/ 案例卡。手动触发。"
disable-model-invocation: true
argument-hint: <轨迹路径或文档路径> [算子名]
---

# triton-knowledge-memory-ingest — 分类1「内存与切块」知识生产

覆盖二级维度 **1.1–1.4**，落点 `runbooks/memory-sram/`（泛化优化点）+ `ops/`（算子案例卡）+ `reference/`（相关官方事实卡）。

## 提取点清单（本分类只收这些主题）

| 维度 | 主题 | 提取点 |
|---|---|---|
| 1.1 Memory Access Efficiency | 访存效率 | GM→UB 连续访存 Coalescing、Burst 搬运合并、UB 内存复用、减少高延迟 HBM/GM 访问 |
| 1.2 Tiling & SRAM Budgeting | 切块与片上预算 | `BLOCK_M/N/K` 自适应划分公式、SRAM/UB 容量溢出约束求解、Tail Tile 余数块处理 |
| 1.3 Data Layout & Format | 数据排布 | ND ↔ NZ 格式转换消除、Transposed Load、Stride 计算优化（Cube 单元强依赖 NZ/Fractal） |
| 1.4 Hardware Memory Alignment | 硬件对齐 | **32-Byte**（VV 场景）与 **512-Byte**（CV 场景）对齐、自动 Padding、Sub-byte/FP8 Pack & Unpack |

## 输入

- **轨迹**（首选）：`<轨迹集占位>`——成功探索的 `report.md` / `summary.json` / `session.jsonl` / `perf_result.json`；失败轨迹同样有价值（AP/CT 原料）。
- **文档**：`<上游文档仓占位>`——官方文档中本分类主题（UB 容量、DMA 对齐、NZ 格式）的页面。
- **golden**：

## 工作流（五阶段）

### 阶段 1：采集

通读输入材料。轨迹重点看：tiling 参数的取值与推导、UB 相关报错（OOM/溢出）、访存/搬运相关的优化动作、对齐相关报错与修法、`summary.json` 的 speedup 与 `per_shape_results`。

### 阶段 2：提取「诊断→修改→验证」三元组

每条候选知识记录三要素：

- **诊断**：问题症状（报错原文 / profiling 现象 / 性能塌点），带出处；
- **修改**：具体动作（BLOCK 调整公式 / 搬运合并 / 布局转换消除 / padding），带代码片段出处；
- **验证**：修改前后的可复现数据（speedup、编译通过、精度通过），**必须指回轨迹的 benchmark 记录，禁止估算**。

### 阶段 3：蒸馏落盘

- 三元组 → runbook 条目：正向 `OPT-*`、反模式 `AP-*`、约束陷阱 `CT-*`（对齐硬校验类多落 CT）。
- 条目按 [`STRUCTURE-runbook.md`](STRUCTURE-runbook.md) 骨架写：**占位名**（`inputA`/`blockM`/`ubBudget`…，禁算子业务名与 golden 变量）、`dimension` 标到二级维度（如 `1.2`）、泛化层级（`Triton通用` / `条件性`+失效边界）、「坏实践（反例）」必填（无则「待补充」）。
- 算子特定事实（该算子的 tiling 方案、UB 预算表、性能数据）→ `ops/` 案例卡，经「实例化 ⇄ 已知实例」与 runbook 双向回链。

### 阶段 4：增量合并

`runbooks/memory-sram/` 已存在同类条目 → **不新建**，只在「已知实例」append 反链；确为新点 → 新增 `OPT-N`（ID 不复用、不重排）。**合并由干净上下文子 Agent 执行**（不内联合并，防单算子细节污染泛化判断）。

### 阶段 5：校验 + 图谱

- 自检：frontmatter（okf.v1 + `dimension`）、锚点闭合、占位名泄漏（grep 算子业务名黑名单）、性能数字有出处。
- 图谱：`python3 ../triton-knowledge-ingest/scripts/okf_graph.py --knowledge-root <知识库根> candidates → judge → inject → viz → verify`（规则见编排器 §4）。

## 关键原则

- **一切结论来自一手材料**：轨迹 benchmark 记录 / 报错原文 / 官方文档固定版本；转述 ≠ 权威。
- **32B 与 512B 场景区分是硬知识**：VV（纯 vector）与 CV（cube↔vector）场景的对齐要求不得混写，条目必须标明适用场景。
- **UB 预算是公式不是感觉**：Tiling 类条目必须给出可代入的容量约束求解式（含 dtype 字节系数），而非"调大/调小试试"。
- **GPU 概念先转换**：coalescing 等概念落库时写成 GM→UB 连续访存语境，不写 Warp/SIMT 措辞。
