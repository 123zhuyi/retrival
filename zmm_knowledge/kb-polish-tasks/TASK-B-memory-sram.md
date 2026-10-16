# 窗口 B 任务书：memory-sram 分类 exp 卡片打磨

## 0. 背景（必读）

Triton-Ascend 知识库（OKF v1）从 `cannbot-skills_5171` 的 raw 经验库批量迁移了 303 条 YAML 经验卡片。批量转换脚本已生成骨架，但存在三个系统性缺口：

1. **「坏实践（反例）」多为「待补充」**（全库 274/303）
2. **「失效边界」全部为「待补充」**（303/303）
3. 部分卡片有**算子业务名残留**（如 Q/K/V、具体算子名），违反 runbook「算子无关」红线

你负责 **memory-sram 分类** 的 **108 个 exp_*.md 卡片**。

## 1. 工作范围

目录：`/home/w00836812/shixi_test02/zmm_knowledge/triton-ascend-kb.bak.20260807/runbooks/memory-sram/`

- 只处理该目录下 `exp_*.md` 文件（约 108 个）
- **不要动**非 exp_ 开头的原有 runbook（如 `vectorized-coalesced-io.md`、`block-ptr-vs-scalar-load.md` 等）——那是窗口 D 的活
- **不要动**其他分类目录

原始 YAML 源文件（参考用，不要修改）：
`/home/w00836812/shixi_test02/cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/raw/experiences/<category>/<exp_id>.yaml`

## 2. 每张卡片的打磨规范

卡片骨架遵循 OKF v1（规范见 `/home/w00836812/shixi_test02/zmm_knowledge/Triton-Ascend-KernelWiki/skills/triton-knowledge-memory-ingest/STRUCTURE-runbook.md`）。memory-sram 分类的特殊纪律：

- **32B 与 512B 场景区分**：VV（纯 vector）与 CV（cube↔vector）场景的对齐要求不得混写
- **UB 预算是公式不是感觉**：涉及 tiling 的条目应给出可代入的容量约束式
- **GPU 概念转换**：coalescing 写成 GM→UB 连续访存语境，不写 Warp/SIMT 措辞

对每张卡片：

### 2.1 补全「坏实践（反例）」

- 从原始 YAML 的 `note`、`optimization_points` 中提炼具体的错误写法 → 后果 → 正解
- 若 YAML 没有反例，根据优化点原理推一个合理反例（如：连续访存优化 → 反例是"stride 访问导致性能损耗 -40%"），不得编造性能数字
- 实在无法提炼时保留「待补充」

### 2.2 补全「失效边界」

格式：`【泛化层级】：<Triton通用|条件性>（迁移条件：<条件>；失效边界：<边界>）`

- 结合 `match_cond`（hardware / core_param / data_type）写具体边界
- 例：`失效边界：地址发散的 gather 场景（此时见 gather-random-address-slow）；BLOCK 超过 UB 容量时失效`

### 2.3 检查占位名

- 正文不出现具体算子业务名（代码块内可保留）
- 维度参数 M/N/K 不要误替换

### 2.4 标题检查

- title 为泛化词时补全具体主题；title、description、正文 `## ` 标题三处一致

### 2.5 禁止事项

- 不改动 `created_at`、`confidence`
- 不新增/删除文件、不改文件名
- 不写摄入/批次口吻
- 不跑图谱/索引脚本（窗口 D 统一跑）

## 3. 工作方式建议

- 分批处理：每批 20-30 个文件
- 用 `grep -l '待补充' runbooks/memory-sram/exp_*.md` 找待办
- 参考同目录已有成熟 runbook（如 `vectorized-coalesced-io.md`）的写法

## 4. 验收标准

- [ ] `失效边界：待补充` 剩余 ≤ 20%
- [ ] `坏实践（反例）：待补充` 剩余 ≤ 30%
- [ ] 正文无业务名残留（代码块除外）
- [ ] frontmatter 可正常解析

## 5. 冲突回避

- 只写 `runbooks/memory-sram/exp_*.md`
- 不 commit、不 push
