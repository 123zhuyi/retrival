# Knowledge Contribution Guide

本文说明外部贡献者如何向独立 triton-ascend 知识库提交 PR、勘误和新增知识。`cannbot-skills` 插件仓只接收工具、流程和规范改动；真实知识正文应提交到独立知识库仓库。

## 可接受的贡献

- `reference/`：triton-ascend 官方文档、API、IR、Profiling、术语的蒸馏卡片。
- `ops/`：基于固定来源（成功案例/轨迹）的算子设计 wiki。
- `runbooks/`：6 大分类的优化点、反模式、约束陷阱、field notes、版本迁移经验：
  - `runbooks/memory-sram/`（分类1 内存与切块）
  - `runbooks/compute-pipeline/`（分类2 计算与流水）
  - `runbooks/algorithm-numerics/`（分类3 算法与数值）
  - `runbooks/portability/`（分类4 规范与移植）
  - `runbooks/autotuning/`（分类5 调优空间设计）
  - `runbooks/meta-skills/`（分类6 Agent 元技能）
- `contrib/`：社区贡献的知识卡片、勘误、补充案例和待验证经验。
- 勘误：修正错误 API、平台限制、Tiling、BLOCK_SIZE、UB 预算、对齐、优化结论或失效边界。
- 治理修复：frontmatter、source、index、graph、search、log、死链、status/confidence。
- 知识编译能力：新增或改进把轨迹/上游材料编译成 OKF 卡片的 skill、adapter 或工具。
- 知识检索能力：新增或改进 `triton-knowledge-query` 检索模式、召回/排序策略、alias/tag/type 映射。
- 工具修复：lint、query、graph、compile 脚本和相关验证方案。

## 不接受的贡献

- 无 source 的高风险 API、平台、精度、性能或默认注入结论。
- 大段复制官方文档、源码、日志或第三方材料。
- 密钥、token、账号、内网地址、私有路径、不可公开日志。
- 只适用于个人环境且无法复现、无法追溯证据的经验。
- 把单个算子的私有技巧误写成通用 runbook 优化点。
- 把 GPU（CUDA/SIMT）语境的概念（如 Warp/Wavefront utilization）不加转换地搬进 Triton-Ascend 知识库。
- 静默删除错误知识；错误知识应优先 `retracted` 或转成负知识。

## PR 必须说明

- 变更类型：新增、更新、勘误、治理修复、知识编译能力、知识检索能力、工具修复。
- 影响范围：涉及哪些内容根、分类 runbook、卡片、图谱、索引或工具。
- 证据来源：官方文档 URL、固定 commit 链接、开发轨迹证据（session/report）或复现实验。
- 处理结论：哪些 applied、deferred、rejected，以及原因。
- 验证结果：列出已运行的 lint、query、graph verify；涉及测试时补充结果。

## 贡献知识编译 skill

知识编译 skill 的职责是把轨迹/上游材料编译成符合 OKF 规范的知识卡片。插件仓只接收编译能力本身，不接收真实大体量知识正文。

新增或改造知识编译能力时，必须满足：

- 先定义输入来源、目标知识分类和落点目录（`reference/`、`ops/`、6 个 `runbooks/<category>/` 之一或 `contrib/`）。
- 产物必须是蒸馏后的 OKF 卡片，不能复制原文、源码或日志。
- 卡片 schema 必须说明 `type`、`resource` / `sources`、`tags`、`dimension` 等字段如何生成。
- 轨迹源必须固定到具体 session/report 路径 + commit；文档源必须能追到稳定 URL 或版本 pin。
- 对已有卡片必须做增量更新，不整体覆盖。
- 实质知识变更必须同步 `index.md`、`graph/`、`log/` 和必要的 query verify。

编译入口统一由 `triton-knowledge-ingest` 路由：

- 如果只是接入新的轨迹形态或文档形态，优先扩展对应分类的已有生产 skill。
- 如果出现新的知识分类或生成链路，可以新增 `skills/<name>/SKILL.md`，并同步更新 `skills/triton-knowledge-ingest/SKILL.md` 的路由表。
- `triton-knowledge-ingest` 只做入口判断和路由编排，不应在其中直接写具体卡片正文。

## 贡献知识检索 skill

知识检索扩展的默认入口是 `triton-knowledge-query`。新增知识分类或新增社区贡献卡片后，应优先扩展检索模式（`modes/`），而不是在主检索逻辑中临时硬编码。

每个检索模式应说明：适用场景、目标知识（目录/type/tags/dimension/status）、召回路线、排序路线、图谱扩展、注入方式和验证场景。

## 单卡准入标准

- frontmatter 完整，至少包含 `type`、`title`、`description`、`resource` 或 `sources`、`tags`、`status`、`confidence`；runbook 条目卡另需 `dimension`（如 `1.2`）。
- git 源必须使用 fixed commit 的 `blob` 或 `tree` URL；轨迹源必须固定到 session/report 路径 + commit。
- 多源卡必须有且只有一个 `primary`，且 `resource == primary.url`。
- 正文必须是蒸馏后的使用知识，不是原文或源码搬运。
- `reference/` 写官方事实；`ops/` 写单算子设计事实；`runbooks/` 只写跨算子可复用规律、失效边界和坏实践；`contrib/` 写社区贡献、勘误、补充案例和待验证经验。
- runbook 条目必须算子无关（占位名骨架），业务实例落 `ops/` 案例卡并经「已知实例」反链。
- 每级 `index.md` 只列本层直接子项。

## 勘误规则

- 勘误必须指出错误位置、错误原因、正确证据和替代结论。
- 过时知识标 `deprecated`，错误知识标 `retracted`，并写明原因、时间和替代关系。
- 撤除正知识时，优先沉淀成负知识（AP-*）或约束陷阱（CT-*），避免后续重复踩坑。
- 平台差异（910B/910C/310P）、triton-ascend 版本差异、精度差异不能互相覆盖，必须写清适用条件。

## 合入门槛

普通知识变更至少通过：

```bash
python3 skills/triton-knowledge-query/scripts/knowledge_query.py --knowledge-root <knowledge-root> verify
python3 skills/triton-knowledge-ingest/scripts/okf_graph.py --knowledge-root <knowledge-root> verify
python3 skills/triton-knowledge-lint/scripts/knowledge_lint.py --knowledge-root <knowledge-root>
```

## Review 分级

- R0：导航、索引、术语。需要 lint 和 link check。
- R1：文档蒸馏事实。需要 source 校验和抽查。
- R2：算子设计事实。需要固定来源（轨迹/源码）核验。
- R3：通用优化点。需要适用条件、失效边界、坏实践和语义抽查。
- R4：API、平台、精度约束。需要强 source 或领域 owner 确认。
- R5：跨任务默认注入知识。需要 source、status、confidence、conflict 和 trace 检查。
