# triton-kb-cann-ingest Skill 收尾报告

> 本报告不是算子入库报告，而是 **skill 构建收尾报告**。任务原意（历史会话）是把"CANN 算子源码分析 → Triton-Ascend 迁移抽象 → KernelWiki 入库"这套 12 节流程固化成一个可复用 Skill。上一轮已完成可行性评估并建好了 skill 主体，本轮完成剩余缺口并清理 KB 验证债务。

## 1. 任务参数

| 参数 | 值 | 状态 |
|---|---|---|
| 目标 | 把 12 节 CANN 静态入库流程做成 Skill | source-confirmed |
| KERNELWIKI_ROOT | `kernelwiki/`（参考用） | source-confirmed |
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer`（仅用于 scaffold 冒烟测试，未做新算子分析） | source-confirmed |
| 模式 | static_ingest（skill 形态） | - |
| 放置位置 | `.claude/skills/triton-kb-cann-ingest/`（原生 Claude Code 位置） | inferred（沿用上一轮 SKILL.md 内既定路径） |

## 2. 可行性结论（沿用上一轮）

**做成 Skill，不做 subagent。** 理由：(1) 12 节深分析有多个需用户过目的中间产物，Skill 跑在主上下文可随时纠偏；(2) 入库需治理（不得假称已验证），Skill 可把"产出候选 → validate → 报告"与"发布"分离；(3) 重活（CANN 源码定位、dispatch/tiling 扫描）可由 Skill 指挥 Explore subagent 并行扇出，主上下文只留结论。

仓库已有两个配套 skill：`triton-kernelwiki-retriever`（只读检索，本 skill 的消费者）与 `triton-kb-source-builder`（单来源 → 候选页 + 治理发布，原子粒度）。本 skill 填的是"整族算子深分析 → 多页 wiki + artifact bundle + 总报告"这一更大粒度的缺口，与两者不重叠。

## 3. Skill 结构（交付物）

```
.claude/skills/triton-kb-cann-ingest/
├── SKILL.md                              # 10 阶段流程 (S1–S10) + static_ingest 治理红线
├── references/
│   ├── migration-rubric.md               # 迁移四分类判定规则 + 表格模板
│   ├── artifact-bundle-layout.md         # PROVENANCE.yaml schema + bundle 文件规范
│   ├── triton-skeleton-templates.md      # elementwise/norm/matmul/attention skeleton 模板
│   └── report-template.md                # 14 节总报告骨架 + 质量清单
└── scripts/
    └── scaffold.py                       # init + update-provenance（本轮新建）
```

SKILL.md 把 12 节任务映射为 10 个阶段（S1 定位源码 → S2 语义 → S3 迁移分类 → S4 tiling/分核 → S5 模板/dispatch → S6 流水 → S7 Triton skeleton → S8 入库 → S9 validate/query → S10 报告），每阶段标注对应的报告章节与产物路径，并强制每条结论可追溯到源码路径+行号。

## 4. 本轮完成的工作

### 4.1 新建 `scripts/scaffold.py`（核心缺口）

SKILL.md 在两处引用 `scaffold.py` 但脚本此前不存在。本轮新建 336 行独立脚本（stdlib + PyYAML，不依赖 `triton_ascend_kb` 包），两个子命令：

| 子命令 | 作用 |
|---|---|
| `init` | 建 `artifacts/<bundle>/` + 起始 `PROVENANCE.yaml`（自动从源码 git 推断 `upstream_repo`/`upstream_sha`/`origin_url`/`retrieved_at`，无法推断则填 `<fill: ...>` 占位）+ `reports/<op>-ingest-report.md` 骨架（从 report-template.md 替换占位符生成）；对不在 `tags.yaml#kernel_types` 的 `--type` 给警告 |
| `update-provenance` | 重算 bundle 内所有文件的 sha256 并重写 `PROVENANCE.files[]`；保留作者已写的 `role`/`mode`，新文件按扩展名给默认 role（.py→derived-source，.md/.yaml→approach-notes） |

冒烟测试（用一次性算子 `smoketest`，测完即清理）验证：
- `init` 正确推断 `upstream_repo: cann/ops-transformer`、`upstream_sha: d28d8f99...`、`origin_url: local://artifacts/ops-transformer`、`derived_from: ops-transformer-smoketest`、`fixture: true`、`files: []`。
- `update-provenance` 对 .md/.py 两个文件计算的 sha256 与独立 `hashlib` 重算**完全一致**（`97c3507a3470...` / `47ee6ef90a11...`），role/mode 分配正确。
- 未知 `--type` 触发 stderr 警告。
- 修复了 `slugify` 把前导下划线转成前导横杠的健壮性 bug（`.strip("-")`）。

### 4.2 修正 SKILL.md 两处措辞瑕疵

1. **治理红线**：原写"reproducibility: `source` for source pages"，但 `source` 不在 `tags.yaml` 词表（`[concept, pseudocode, snippet, runnable, benchmarked]`），且金标准 FA 源页根本不带 `reproducibility` 字段。改为：wiki `reproducibility` 必须用词表值，源页不带该字段。
2. **canonical reference 提示**：SKILL.md 把 `flash-attention-score` 与 `elementwise-abs` 都列为"match their shape"，但 `elementwise-abs` 实际用了非词表值（`confidence: high` / `reproducibility: source`，因 `validate.py` 不校验这两个字段的词表而蒙混过关）。补一句：frontmatter **值**以 `flash-attention-score` 为金标准，`elementwise-abs` 的结构可参考但其 confidence/reproducibility 值不可照抄。

### 4.3 清理历史会话遗留的 KB 验证债务

接手时 `validate.py` 失败 7 处，全是历史会话未完工的 sparse-attention 入库残留：

| 文件 | 问题 | 修复 |
|---|---|---|
| `wiki/techniques/sparse-attention-pipeline.md` | 缺 `sources`；`artifact_dir` 错指源码仓库内无 PROVENANCE 的目录 | 加 `sources: [ops-transformer-sparse-flash-attention]`；移除 `artifact_dir` |
| `wiki/techniques/sparse-attention-tiling.md` | 缺 `prerequisites`、`sources`；同上 `artifact_dir` | 补 `prerequisites` + `sources`；移除 `artifact_dir` |
| `wiki/techniques/sparse-kv-gather.md` | 缺 `prerequisites`、`sources`；同上 `artifact_dir` | 补 `prerequisites` + `sources`；移除 `artifact_dir` |
| `sources/ops-transformer/sparse-flash-attention.md` | 不存在（3 个 sparse 页无可引用的源页） | **新建**最小来源锚点页（repo/commit/path/license/衍生引用/复现性说明） |

新建的 sparse 源页是**来源追溯锚点，不是新分析**--分析内容已在 3 个技巧页里，源页只记录仓库定位（`kunpengcompute/CANN-ops-transformer`，与同仓库的 FA 源页一致）、固定提交 `d28d8f99...`、算子路径 `attention/sparse_flash_attention/`，以及被引用的关键文件行号区域指针。正文中残留的 `artifact_dir` 引用也一并改为指向新源页。`artifact_dir` 被移除是因为这 3 页没有派生 artifact 文件（分析都在 wiki 正文），指向源码仓库目录是错误用法。

## 5. KB 入库页面（本次涉及）

| 路径 | 类型 | id | 状态 |
|---|---|---|---|
| `sources/ops-transformer/sparse-flash-attention.md` | source | `ops-transformer-sparse-flash-attention` | 新建 |
| `wiki/techniques/sparse-attention-pipeline.md` | technique | `technique-sparse-attention-pipeline` | 修复 frontmatter + 正文 |
| `wiki/techniques/sparse-attention-tiling.md` | technique | `technique-sparse-attention-tiling` | 修复 frontmatter + 正文 |
| `wiki/techniques/sparse-kv-gather.md` | technique | `technique-sparse-kv-gather` | 修复 frontmatter |
| `.claude/skills/triton-kb-cann-ingest/scripts/scaffold.py` | (skill 脚本) | - | 新建 |
| `.claude/skills/triton-kb-cann-ingest/SKILL.md` | (skill 文档) | - | 修正 2 处措辞 |

## 6. validate / generate-indices / query 执行结果

环境可运行脚本，全部实跑：

```
$ python kernelwiki/scripts/validate.py
OK: KernelWiki validation passed                                   # 接手时 7 ERROR，现 0

$ python kernelwiki/scripts/generate-indices.py
Generated queries/{by-problem,by-technique,by-hardware-feature,by-repo,by-kernel-type}.md

$ python kernelwiki/scripts/query.py flash_attention_score --compact        # 3 hits: source/kernel/pattern
$ python kernelwiki/scripts/query.py attention tiling --compact             # 10 hits，含 3 个 sparse 技巧页
$ python kernelwiki/scripts/query.py attention mask --compact               # 10 hits
$ python kernelwiki/scripts/query.py attention triton ascend --compact      # 10 hits（Triton-Ascend 文档）
$ python kernelwiki/scripts/query.py sparse_flash_attention --compact       # 2 hits，含新建 sparse 源页 ✓
```

关键验证：新建的 sparse 源页可被 `query.py` 检索到；3 个修复后的 sparse 技巧页出现在 `attention tiling` / `attention mask` 结果中。检索契约端到端正常。

## 7. 如何使用该 Skill

```bash
# 1. 脚手架：建 bundle + PROVENANCE + 报告骨架
python .claude/skills/triton-kb-cann-ingest/scripts/scaffold.py init \
  --root kernelwiki --source kernelwiki/artifacts/ops-transformer \
  --operator flash_attention_score --type attention --variants forward,inference

# 2. 按 SKILL.md S1–S7 做静态分析，把派生文件写入 artifacts/<bundle>/

# 3. 重算 sha256
python .claude/skills/triton-kb-cann-ingest/scripts/scaffold.py update-provenance \
  --root kernelwiki --bundle flash-attention-score

# 4. 写 source 页 + wiki 页（frontmatter 见 SKILL.md S8），然后校验
python kernelwiki/scripts/validate.py
python kernelwiki/scripts/generate-indices.py
```

## 8. 未验证 / 后续事项

| 项 | 说明 |
|---|---|
| sparse 源页的 `.../` arch 子目录路径 | 源页表格用 `.../` 简记 arch32/arch35/regbase 子目录，具体路径需在固定提交下确认，已标 `needs_verification` |
| scaffold.py 未跑真实算子入库 | 只做了 `smoketest` 冒烟（已清理），未用它端到端入库一个新算子。建议下次入库一个新算子时做一次完整 dry-run |
| ~~skill 放置位置~~ | ✅ 已统一部署：规范副本移至 `integrations/cannbot-skills/ops/triton-kb-cann-ingest/`（与另两个 skill 同位），symlink 回 `.claude/skills/` 保持本仓库可用，补 `agents/openai.yaml`。详见 §10 |
| `elementwise-abs` 非合规 frontmatter | 该页 `confidence: high` / `reproducibility: source` 非词表值，因 validate 不校验而通过。本轮未改它（超出范围），仅在 SKILL.md 加了警示。如需 KB 全量合规可后续修正 |
| `tags.yaml` / `queries/*.md` 改动 | `tags.yaml` 与 4 个 `queries/by-*.md` 在接手前已被历史会话改动；`queries/` 由本轮 `generate-indices.py` 重新生成。均非本轮主动编辑 |

## 9. 修改 / 新建文件清单

**新建：**
- `.claude/skills/triton-kb-cann-ingest/scripts/scaffold.py` — skill 脚手架脚本（init + update-provenance）
- `kernelwiki/sources/ops-transformer/sparse-flash-attention.md` — SparseFlashAttention 来源锚点页
- `reports/triton-kb-cann-ingest-skill-report.md` — 本报告

**修改：**
- `.claude/skills/triton-kb-cann-ingest/SKILL.md` — 修正 reproducibility 红线 + canonical reference 警示
- `kernelwiki/wiki/techniques/sparse-attention-pipeline.md` — frontmatter（sources，移除 artifact_dir）+ 正文引用
- `kernelwiki/wiki/techniques/sparse-attention-tiling.md` — frontmatter（prerequisites + sources，移除 artifact_dir）+ 正文引用
- `kernelwiki/wiki/techniques/sparse-kv-gather.md` — frontmatter（prerequisites + sources，移除 artifact_dir）
- `kernelwiki/queries/by-{problem,technique,hardware-feature,kernel-type}.md` — `generate-indices.py` 重新生成

## 10. 更新：统一部署（本轮追加）

应要求把 skill 部署模型与另两个 skill 统一：

- **规范副本迁移**：`.claude/skills/triton-kb-cann-ingest/` -> `integrations/cannbot-skills/ops/triton-kb-cann-ingest/`（与 `triton-kernelwiki-retriever`、`triton-kb-source-builder` 同位）。
- **symlink 回挂**：`.claude/skills/triton-kb-cann-ingest` -> `../../integrations/cannbot-skills/ops/triton-kb-cann-ingest`（相对路径，可移植），skill 在本仓库仍可直接调用。
- **结构对齐**：补 `agents/openai.yaml`（cannBot 接口描述，与另两个 skill 一致）。

验证：
- `find -L .claude/skills/triton-kb-cann-ingest -type f` 列出全部 7 个文件（含新增 `agents/openai.yaml`）。
- `python .claude/skills/triton-kb-cann-ingest/scripts/scaffold.py --help` / `init --help` 经 symlink 路径正常。
- `scaffold.py` 内 `Path(__file__).resolve()` 正确跟随 symlink 到 integrations/ 真实路径，`REPORT_TEMPLATE` 可定位。
- `python kernelwiki/scripts/validate.py` 仍 OK（skill 迁移未影响 KB）。

最终 skill 结构：
```
integrations/cannbot-skills/ops/triton-kb-cann-ingest/   # 规范副本
├── SKILL.md
├── agents/openai.yaml                  # 本轮补齐
├── references/{migration-rubric,artifact-bundle-layout,triton-skeleton-templates,report-template}.md
└── scripts/scaffold.py
.claude/skills/triton-kb-cann-ingest    # -> ../../integrations/cannbot-skills/ops/triton-kb-cann-ingest
```
