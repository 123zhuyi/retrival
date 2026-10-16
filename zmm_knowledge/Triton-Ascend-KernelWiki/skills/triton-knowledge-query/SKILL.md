---
name: triton-knowledge-query
description: "Use when 用户正在处理昇腾 NPU 上的 Triton/Triton-Ascend 算子任务，并询问 tl.* API 名称或可用性、参数语义、tiling/BLOCK_SIZE、UB/GM、ND/NZ、对齐、多核 grid、autotune、混合精度、数值稳定、平台/版本支持、编译/运行错误、精度、性能或 Profiling；在搜索本地 triton-ascend 安装包或源码前触发。"
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

# triton-knowledge-query — Triton-Ascend 知识检索与证据前置

本 skill 是 Triton-Ascend / 昇腾 NPU 算子知识库的只读检索入口。它负责把用户问题转成可检索的短 query，查找 reference API、ops 算子案例、6 大分类 runbooks 优化经验和 field notes，补充图谱邻居，再把证据交给后续回答、设计、编码、调试和性能优化使用。

安装后的 skill 入口是本 skill 内的 `scripts/knowledge_query.py`（纯 stdlib，确定性默认）。Agent 调用本 skill 后，优先用 `preflight` 形成短 query、检索路线和第一批证据，再由 Agent 读取卡片、判断充分性，并按任务类型组合 `overview/pipeline/grep/get/neighbors`。

> 知识库 root 解析顺序：`--knowledge-root` / `--knowledge-roots`、`TRITON_ASCEND_KNOWLEDGE_ROOT(S)`、`OKF_KNOWLEDGE_ROOT(S)`、`~/.config/cannbot/triton-knowledge.env`、有限结构探测。

参数提示：`preflight --task "<当前任务/报错/疑问>"`、短 `--query`、`--method/--scope/--kind` 等检索参数，或 `<卡片 doc-id>`。

## What this skill solves
- **API/语义查证**：确认 Triton/Triton-Ascend API（tl.load/tl.store/tl.dot/tl.reduce/tl.arange 等）、参数语义、dtype/shape、平台/版本约束。
- **算子实现设计**：查找 tiling/BLOCK_M/N/K 方案、UB 预算、grid 与多核切分、double buffering、autotune 空间设计经验。
- **数据搬运与内存问题**：定位 GM→UB 访存、ND↔NZ 格式、32B/512B 对齐、stride 计算、padding 等约束和样例。
- **并行与计算模式**：检索多核负载均衡、coreDim 限制、融合（prologue/epilogue）、reduction、atomic 等实现模式和反模式。
- **调试与诊断**：根据编译报错、运行错误、精度异常、性能异常、msprof/Profiling 线索查找诊断卡、约束说明和已知修复路径。
- **相似样例与经验复用**：查找 ops/ 下相似算子案例、runbooks/ 下 6 大分类优化经验、reference/ 下 API 依据，并通过图谱邻居扩展上下文。
- **依据与出处补全**：为方案结论、代码修改、性能判断和问题定位补充 doc-id、resource、相关卡片和上游来源。

## Trigger scenarios
- 用户问"怎么写、怎么实现、怎么调、怎么优化、哪里有样例、这个 API 什么含义、这个报错怎么处理"时，只要主题属于 Triton-Ascend、昇腾 NPU、AICore kernel 或算子开发，先调用本 skill。
- 用户提到或暗含以下关键词时优先触发：Triton、triton_ascend、`@triton.jit`、tl.load/tl.store、BLOCK_SIZE、tiling、UB、GM、ND/NZ、对齐、grid、多核、coreDim、double buffer、num_stages、autotune、mask、broadcast、div/mod、混合精度、BF16/FP8、数值稳定、reduction、atomic、tl.multiple_of、910B/910C/310P、msprof、Profiling、精度、性能、编译失败、运行失败、相似样例。
- 开始方案设计、代码实现、代码修改、代码审查或调试前先做知识库 `preflight`；从设计转实现、从编译转运行、从正确性转性能时再次检索。
- 对某个概念、API、错误码、配置项、实现路径、最佳实践或反模式不确定时，先查知识库再推断。
- 不触发：纯机械编辑、格式化、文件移动、用户明确要求不查知识库、与 Triton-Ascend 算子知识无关的通用问答、生成新知识卡（应交给 `triton-knowledge-ingest` 路由到对应分类生产 skill）。

## API 证据顺序
Triton-Ascend API 查证固定执行 `知识库 -> 固定版本上游 -> 本地安装包`：
1. **知识库**：先运行 `preflight`，逐张读取 `read_first`，并按需用 `get --section` 或 `local_path` 深读。
2. **固定版本上游**：读卡后仍缺少精确签名、平台/版本约束或实现细节，再沿卡片 `resource` / `sources` 核对固定 revision 的官方文档或源码。
3. **本地安装包**：只有知识库未命中、前两层仍不足，或任务明确要求核验当前已安装 triton-ascend 的实际行为时，才检查本地安装。检查时同时确认并记录安装版本、目标平台和读取路径，并与知识卡或固定版本上游交叉核对；冲突按版本/平台差异处理并保留两侧依据。

## Default workflow
1. 运行 `knowledge_query.py discover` 确认选中的 knowledge base root；root 不对时显式加 `--knowledge-root <知识库根>`。
2. 在实质性推理、计划或改代码前运行 `knowledge_query.py preflight --task "<当前任务/报错/疑问>"`；上下文紧张时加 `--brief`。
3. 先读取 `route`、`read_first`、`relevance`、`read_purpose`、`verify_after_read`、`next_actions`、`follow_candidates`、`missing_signals`、`sufficiency_rule`。不要把 `results` 当最终答案。
4. 如果 `route=browse_first`，先运行 `overview`/`browse` 建立目录视图；如果 `route=search_first`，先 `get` 或 `Read local_path` 读取 `read_first` 中 1-3 张卡片。
5. 输出中的 `path` 是稳定 doc-id；`local_path` 可直接 `Read` 做多轮阅读。后续检索继续使用 `preflight` 生成的命令。
6. 读取卡片后由 Agent 判断证据是否满足 `sufficiency_rule`：至少 1 张 strong 证据卡，或 2 张 medium 互撑；weak 不能单独支撑结论。不满足时从 `follow_candidates` 选择 `neighbors`、`pipeline`、`grep` 或二次 `preflight`。
7. 只有证据足够后再回答、设计或修改代码；结论中保留 doc-id、`resource`。

## How to query
```
knowledge_query.py preflight --task "<当前任务/报错/疑问>" [--brief]
knowledge_query.py plan      --task "<当前任务/报错/疑问>"
knowledge_query.py overview  [--task T|--query Q] [--scope S] [--groups N]
knowledge_query.py search   --query Q [--scope/--dir S] [--kind/--category]     # 确定性 BM25F
knowledge_query.py discover
knowledge_query.py recall   --method bm25|tfidf|tagtype|graph|dense [facets]
knowledge_query.py rerank   --method bm25f|tagidf|quality|reranker|llm-judge --hits -
knowledge_query.py pipeline --recall bm25,tagtype --rerank bm25f --query Q [facets]
knowledge_query.py grep     <regex> [--scope] [--only body|frontmatter|all]
knowledge_query.py get      <doc-id>... [--section H] [--max-chars N]
knowledge_query.py neighbors <doc-id> --hops N
knowledge_query.py verify [--level index|schema|strict] [--limit N] [--json]
knowledge_query.py eval [--cases FILE] [-k N] [--fail-under FLOAT]
```
`search` 过滤参数：`--scope/--dir`（接受 `reference/`、`ops/`、`runbooks/`、`runbooks/memory-sram/` 等分类子目录；`all` 不限）。`recall/pipeline` 支持完整 facets：`--type --tags --paradigm --severity --confidence --status active|verified|stub|all`。

## Query construction / 检索原则（与 Asc 版一致）
- **先跑 preflight**：开始需要领域知识支撑的实现、调试、设计、优化或验收前运行；先看 `route/read_first/next_actions/follow_candidates/missing_signals/sufficiency_rule`。
- **让 Agent 做决策**：脚本只给候选；Agent 必须读卡、判断证据充分性（≥1 strong 或 ≥2 medium 互撑）。
- **一次 search 一个意图**：不要把完整任务描述塞进单个 `--query`，不要混合多个 API 猜测；多路证据用 `preflight`/`pipeline`。
- **先收窄再召回**：用 `--scope/--kind/--status` 缩小范围，别盲目调大 `-k`。
- **精确符号再 grep**：对 API、错误码、关键配置项用 `grep` 复核，再 `get` 注入整卡。
- **召回求全，重排默认确定性**：词法 `bm25`（别名默认开）、结构 `tagtype`、关系 `graph`；模型路线（dense/reranker/llm-judge）opt-in、按指纹缓存、未配置后端返回结构化错误。
- **默认排 stub**（`--status active`）；找待补卡用 `--status stub/all`。
- **图谱多跳是核心**：选卡后 `neighbors` 沿 typed edge 取周边，给连通的知识面。
- **先片段再整卡**：最终引用各卡 doc-id + `resource`；`local_path` 只用于本地读取。

## Route selection
- API/符号语义：`preflight` 后 `search --scope reference/`，再 `grep` 精确复核符号。
- 相似样例：`pipeline --recall bm25,tagtype --rerank bm25f --scope ops/`，再 `neighbors`。
- 性能/tiling/优化：`pipeline --recall bm25,tagtype --rerank quality --scope runbooks/`（可进一步收窄到 `runbooks/<category>/`），结果不足时扩到 `ops/`。
- 报错/日志诊断：先 `preflight`，对错误码、关键短语跑 `grep`，再 `get` 命中的诊断卡或相似案例。
- 元技能（诊断树/修复链/动作时序）：`search --scope runbooks/meta-skills/`。

## 检索模式（recipes，可扩展）
- [`modes/similar_examples.md`](modes/similar_examples.md) — 相似样例（ops/，tag/type/paradigm）
- [`modes/base_api.md`](modes/base_api.md) — 基础 API（reference/，bm25+kind）
- [`modes/optimization_practice.md`](modes/optimization_practice.md) — 优化实践（runbooks/+ops/，quality 重排）
- 新增模式：在 `modes/` 丢一份 `.md`（模板见 `modes/README.md`）。

## 输出 / 引用
答案给出依据卡的 doc-id + `resource`；标注 confidence。检索证据不足时明确说明已查过哪些 query/grep。
