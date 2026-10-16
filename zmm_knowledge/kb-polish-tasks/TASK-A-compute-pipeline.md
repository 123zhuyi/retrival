# 窗口 A 任务书：compute-pipeline 分类 exp 卡片打磨

## 0. 背景（必读）

Triton-Ascend 知识库（OKF v1）从 `cannbot-skills_5171` 的 raw 经验库批量迁移了 303 条 YAML 经验卡片。批量转换脚本已生成骨架，但存在三个系统性缺口：

1. **「坏实践（反例）」多为「待补充」**（全库 274/303）
2. **「失效边界」全部为「待补充」**（303/303）
3. 部分卡片有**算子业务名残留**（如 Q/K/V、具体算子名），违反 runbook「算子无关」红线

你负责 **compute-pipeline 分类** 的 **148 个 exp_*.md 卡片**。

## 1. 工作范围

目录：`/home/w00836812/shixi_test02/zmm_knowledge/triton-ascend-kb.bak.20260807/runbooks/compute-pipeline/`

- 只处理该目录下 `exp_*.md` 文件（约 148 个）
- **不要动**非 exp_ 开头的原有 runbook（如 `scalar-loop-to-dot.md`、`pad-slot-early-exit.md` 等）——那是窗口 D 的活
- **不要动**其他分类目录（memory-sram / portability / algorithm-numerics / autotuning / meta-skills）——那是窗口 B/C 的活

原始 YAML 源文件（参考用，不要修改）：
`/home/w00836812/shixi_test02/cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/raw/experiences/<category>/<exp_id>.yaml`

## 2. 每张卡片的打磨规范

卡片骨架遵循 OKF v1（规范见 `/home/w00836812/shixi_test02/zmm_knowledge/Triton-Ascend-KernelWiki/skills/triton-knowledge-compute-ingest/STRUCTURE-runbook.md`）。对每张卡片：

### 2.1 补全「坏实践（反例）」

- 从原始 YAML 的 `note`、`optimization_points` 中提炼一个具体的错误写法 → 后果 → 正解
- 若 YAML 确实没有反例信息，根据该优化点的原理**推一个合理的反例**（如：块大小优化 → 反例是"盲选 BLOCK_SIZE=4096 导致 UB 溢出"），但不得编造性能数字
- 实在无法提炼时保留「待补充」，不要为了填满而写空话

### 2.2 补全「失效边界」

格式：`【泛化层级】：<Triton通用|条件性>（迁移条件：<条件>；失效边界：<边界>）`

- 结合 `match_cond`（hardware / core_param / data_type / optimization_level）写出具体边界
- 例：`失效边界：BLOCK_SIZE 超过 UB 预算时失效；小 shape (<1024 元素) 无收益`

### 2.3 检查占位名

- 正文中不应出现具体算子业务名（Q/K/V → inputA/inputB/inputC、FlashAttention → attention kernel 等泛化表述）
- **但代码块内**的变量名可保留原样（代码是实例证据）
- 维度参数 M/N/K 不要误替换

### 2.4 标题检查

- 若 title 是泛化词（如 `Scheduling Optimization`），从 optimization_points 第一个 key 补全为 `Scheduling Optimization: grid stride loop` 这类形式
- title、description、正文 `## ` 标题三处保持一致

### 2.5 禁止事项

- 不改动 frontmatter 的 `created_at`、`confidence`（confidence 校准是窗口 D 的活）
- 不新增/删除文件、不改文件名
- 不在正文写摄入/批次口吻（"本次迁移"这类话）
- 不跑 `okf_graph.py` / `knowledge_query.py`（统一由窗口 D 最后跑）

## 3. 工作方式建议

- 分批处理：每批 20-30 个文件，读完对应 YAML 源再改
- 用 `grep -l '待补充' runbooks/compute-pipeline/exp_*.md` 找待办
- 完成后用 `grep -c '待补充'` 自查剩余量

## 4. 验收标准

完成后该目录下 exp_*.md 应满足：

- [ ] `失效边界：待补充` 剩余 ≤ 20%（无法推断的才保留）
- [ ] `坏实践（反例）：待补充` 剩余 ≤ 30%
- [ ] 无 Q/K/V 等业务名残留在正文（代码块除外）
- [ ] frontmatter 未被破坏（`python3 -c "import yaml,sys; yaml.safe_load(open(f).read().split('---')[1])"` 可解析）

## 5. 冲突回避

- 只写 `runbooks/compute-pipeline/exp_*.md`
- 不 commit、不 push（统一最后处理）
