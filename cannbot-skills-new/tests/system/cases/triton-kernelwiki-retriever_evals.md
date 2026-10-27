---
skill_name: triton-kernelwiki-retriever
eval_mode: text
---

# Case 1: 设计阶段触发检索

## Config
- Max Tokens: 80000

## Prompt

我正在用 triton-op-generator 设计一个 LayerNorm 算子，已经做完模板 precheck，接下来应该怎么利用 KernelWiki 帮助设计草图？

## Expected Output

回复应说明在设计阶段调用 `triton-kernelwiki-retriever` 的 `design` 模式，执行 `python .claude/skills/triton-kernelwiki-retriever/scripts/retrieve.py --mode design ...`，产物为 `{工作目录}/knowledge/design_context.json`。应说明将 `official` lane 作为 inspirations、`wiki-kernel`/`wiki-technique` lanes 作为 history_context、`hardware` lane 作为架构上下文传给 `triton-op-designer`。

## Expectations

- [contains] design
- [contains] design_context.json
- [contains] triton-kernelwiki-retriever
- [contains] history_context

---

# Case 2: 修复阶段触发检索

## Config
- Max Tokens: 80000

## Prompt

triton-op-generator 在 Phase 3 验证 LayerNorm 时出现了精度不匹配，Conductor 判断是 A 类错误。我该怎么用 KernelWiki 找修复思路？

## Expected Output

回复应说明在修复阶段调用 `triton-kernelwiki-retriever` 的 `repair` 模式，传入 `--verifier-error-file` 或 `--verifier-error`，产物为 `{工作目录}/knowledge/repair_iter_N.json`。应将结果作为 `repair_knowledge_context` 与 `previous_code`、`verifier_error`、`conductor_suggestion` 一起传给 `triton-op-coding`，并将检索结果视为假设、必须重新跑全部 verifier cases。

## Expectations

- [contains] repair
- [contains] repair_iter
- [contains] repair_knowledge_context
- [contains] verifier_error

---

# Case 3: 优化阶段触发检索

## Config
- Max Tokens: 80000

## Prompt

LayerNorm 算子已经通过精度验证，进入 Phase 4 性能优化。我想先看看 KernelWiki 里有没有相关的优化技巧，应该怎么做？

## Expected Output

回复应说明在优化阶段调用 `triton-kernelwiki-retriever` 的 `optimize` 模式，产物为 `{工作目录}/knowledge/optimize_iter_N.json`。应将非空 lane 作为 `optimization_knowledge_context` 与基线代码、实测性能数据一起传给 `triton-latency-optimizer`，且优化必须经过 correctness 和 benchmark 验证。

## Expectations

- [contains] optimize
- [contains] optimize_iter
- [contains] optimization_knowledge_context
- [contains] benchmark

---

# Case 4: 依赖检查与降级行为

## Config
- Max Tokens: 60000

## Prompt

我运行 triton-op-generator 时提示找不到 triton-kernelwiki-retriever，需要配置什么？

## Expected Output

回复应说明需要满足以下任一条件：安装 `triton_ascend_kb` Python 包（在 KernelWiki 目录执行 `pip install -e .`），或设置环境变量 `TRITON_ASCEND_KB_ROOT=<kernelwiki>/kernelwiki`。若都未满足，应说明 `triton-op-generator` 会告警并跳过 KernelWiki 检索，继续原流程。

## Expectations

- [contains] TRITON_ASCEND_KB_ROOT
- [contains] triton_ascend_kb
- [contains] 跳过
- [contains] KernelWiki
