---
name: triton-knowledge-portability-ingest
description: "分类4 规范与移植（Portability & System Bounds）知识生产 skill：蒸馏动态 shape 泛化、跨版本 API 兼容、编译器约束与 hints、910B/910C/310P 跨芯片适配知识，产出 runbooks/portability/ 优化点与 reference/ 事实卡。手动触发。"
disable-model-invocation: true
argument-hint: <轨迹路径或文档路径> [主题]
---

# triton-knowledge-portability-ingest — 分类4「规范与移植」知识生产

覆盖二级维度 **4.1–4.3**，落点 `runbooks/portability/` + `reference/` + `ops/`。

## 提取点清单（本分类只收这些主题）

| 维度 | 主题 | 提取点 |
|---|---|---|
| 4.1 Correctness & Portability | 正确性与可移植 | 动态 Shape 泛化支持、跨 Triton-Ascend 版本 API 兼容性（保证不同输入规模下的鲁棒性） |
| 4.2 Compiler & Backend Constraint Handling | 编译器约束 | `tl.multiple_of`/`tl.max_contiguous` 提示插入、Triton-Ascend 临时 Bug 的 Workaround、环境变量开关（配合并引导编译器生成高效 CCEC/IR） |
| 4.3 Cross-Hardware Adaptation | 跨芯片适配 | 910B / 910C / 310P 芯片代际能力探测与差异化分发（UB 容量、Vector 宽度差异） |

## 输入

同编排器 §2。轨迹重点看：编译报错与 workaround、版本迁移记录、多平台验证记录、动态 shape 相关修复。

## 工作流（五阶段）

### 阶段 1：采集

轨迹重点看：编译报错原文与修复、hints 插入动作、平台分支（`soc_version`/arch 判断）、动态 shape 泛化改动。

### 阶段 2：提取「诊断→修改→验证」三元组

- **诊断**：编译报错（含完整错误信息）、特定平台失败、特定 shape 失败；
- **修改**：hint 插入、workaround、平台能力探测分支、泛化改写；
- **验证**：通过的平台/版本/shape 矩阵记录。

### 阶段 3：蒸馏落盘

- 三元组 → `OPT-*` / `AP-*` / `CT-*`。**本分类 CT 密度最高**：编译器限制、平台能力差异、版本行为差异多为约束陷阱。
- **适用条件四元组必填**：平台（910B/910C/310P）、triton-ascend 版本、dtype、shape 范围——四者缺一不可，防止跨平台/跨版本互相覆盖。
- 条目按 [`STRUCTURE-runbook.md`](STRUCTURE-runbook.md) 骨架，`dimension`（4.1–4.3）。
- 官方 API 事实（含版本矩阵）→ `reference/` 卡。

### 阶段 4：增量合并

同分类条目已存在 → 只 append「已知实例」或补充平台/版本适用条件；新点 → 新增 ID。**干净上下文子 Agent 合并**。

### 阶段 5：校验 + 图谱

自检 → `okf_graph.py candidates → judge → inject → viz → verify`。

## 关键原则

- **workaround 必须带版本边界**：写清"哪个版本需要、哪个版本已修复"，并给修复后的回退路径（移除 workaround 的条件）。
- **平台差异不互相覆盖**：910B 成立的结论不得写成像对 910C/310P 也成立；不确定的标 `unverified`。
- **hints 给判据不给口诀**：`tl.multiple_of` 等提示的插入条目须写清"什么条件下插入有收益、什么条件下反而约束编译器"。
