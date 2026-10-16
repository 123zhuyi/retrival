# 窗口 C 任务书：portability / algorithm-numerics / autotuning / meta-skills 分类 exp 卡片打磨

## 0. 背景（必读）

Triton-Ascend 知识库（OKF v1）从 `cannbot-skills_5171` 的 raw 经验库批量迁移了 303 条 YAML 经验卡片。批量转换脚本已生成骨架，但存在三个系统性缺口：

1. **「坏实践（反例）」多为「待补充」**（全库 274/303）
2. **「失效边界」全部为「待补充」**（303/303）
3. 部分卡片有**算子业务名残留**，违反 runbook「算子无关」红线

你负责 **4 个较小分类**，共 **47 个 exp_*.md 卡片**：

| 目录 | 文件数（约） |
|---|---|
| `runbooks/portability/` | 33 |
| `runbooks/meta-skills/` | 8 |
| `runbooks/algorithm-numerics/` | 4 |
| `runbooks/autotuning/` | 2 |

## 1. 工作范围

根目录：`/home/w00836812/shixi_test02/zmm_knowledge/triton-ascend-kb.bak.20260807/runbooks/`

- 只处理上述 4 个目录下的 `exp_*.md` 文件
- **不要动**非 exp_ 开头的原有 runbook——那是窗口 D 的活
- **不要动** compute-pipeline / memory-sram——那是窗口 A/B 的活

原始 YAML 源文件（参考用，不要修改）：
`/home/w00836812/shixi_test02/cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/raw/experiences/<category>/<exp_id>.yaml`

## 2. 各分类特殊纪律

- **portability**（规范：`Triton-Ascend-KernelWiki/skills/triton-knowledge-portability-ingest/STRUCTURE-runbook.md`）：
  - **适用条件四元组必填**：平台（910B/910C/310P）、triton-ascend 版本、dtype、shape 范围
  - workaround 类条目须写清版本边界（哪个版本需要、哪个版本已修复）
  - 910B 成立的结论不得写得像对 910C/310P 也成立，不确定标 `unverified`
- **algorithm-numerics**：精度结论必须有误差指标（MERE/MARE/ULP）；精度收益与性能代价成对记录
- **autotuning**：剪枝规则须是可执行判据（如"BLOCK_K × dtypeBytes 须整除 32B"），不收"适当减少候选"类空话
- **meta-skills**：每条须含触发条件 + 动作，不收"要系统性思考"类空话

## 3. 每张卡片的打磨规范

同窗口 A/B：

1. 补全「坏实践（反例）」——从 YAML 的 `note` / `optimization_points` 提炼，或依原理推合理反例，不得编造性能数字
2. 补全「失效边界」——结合 `match_cond` 写具体边界
3. 检查占位名——正文无业务名残留（代码块除外）
4. 标题检查——泛化 title 补全具体主题，三处（frontmatter title/description/正文 ##）一致

禁止事项：不改 `created_at`/`confidence`；不增删文件、不改名；不写批次口吻；不跑图谱/索引脚本。

## 4. 工作方式建议

- 按目录顺序处理：portability → meta-skills → algorithm-numerics → autotuning
- 用 `grep -l '待补充' runbooks/{portability,meta-skills,algorithm-numerics,autotuning}/exp_*.md` 找待办

## 5. 验收标准

- [ ] 4 个目录 `失效边界：待补充` 剩余 ≤ 20%
- [ ] `坏实践（反例）：待补充` 剩余 ≤ 30%
- [ ] portability 条目四元组（平台/版本/dtype/shape）尽量补全
- [ ] frontmatter 可正常解析

## 6. 冲突回避

- 只写上述 4 个目录的 `exp_*.md`
- 不 commit、不 push
