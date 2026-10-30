# Triton-Ascend KernelWiki Demo

本项目以 [mit-han-lab/KernelWiki](https://github.com/mit-han-lab/KernelWiki) 的知识组织、检索脚本和页面契约为基础，为 Triton-Ascend 算子生成 Agent 提供来源可追溯的知识检索。基础版不引入独立重排器、查询缓存、经验评分或自定义记忆协议。

## 1. 对标结论

当前版本已在“知识库使用契约”层面对齐 KernelWiki：

| 能力 | KernelWiki | 本项目 |
|---|---|---|
| 根目录 | `SKILL.md/data/sources/wiki/queries/artifacts/scripts` | 同构 |
| 页面查询 | `scripts/query.py` | 同名脚本、同参数、同评分规则 |
| 单页读取 | `scripts/get_page.py` | 支持 `body/frontmatter/follow-sources/include-code` |
| 全文检索 | `scripts/grep_wiki.py` | 支持 `only/context/any/limit/files-only/ext` |
| 页面来源 | Wiki frontmatter 的 `sources` | 同名字段并校验引用 |
| 代码制品 | `artifact_dir` 与 `PROVENANCE.yaml` | 同构，额外核验文件 SHA-256 |
| Schema 与词表 | `data/schemas.yaml`、`data/tags.yaml` | 同构并替换为 Ascend 词表 |
| 交叉索引 | 六类 `queries/*.md` | 同名生成脚本与索引 |
| 状态统计 | `scripts/repo_status.py` | 同名脚本 |

Triton-Ascend 必要适配只有三项：架构和技术标签替换为 Ascend 词表；新增 `sources/runs/` 表达本地编译、正确性和 profiler 复现实验；保留 `bundle` 薄适配器将 KernelWiki 页面交给算子生成 Agent。该适配器复用原始查询分数，不产生第二套排序。

当前版本尚未对标 KernelWiki 的大规模数据采集自动化，例如真实 GitHub PR 批量抓取、候选台账刷新、版本声明注册表和 core-PR 闭包。这些属于后续数据工程，不影响基础查询契约。

## 2. 目录结构

```text
kernelwiki/
├── SKILL.md
├── index.md
├── scripts/
│   ├── query.py
│   ├── get_page.py
│   ├── grep_wiki.py
│   ├── ingest_triton_ascend_docs.py
│   ├── retag_triton_ascend_docs.py
│   ├── validate.py
│   ├── generate-indices.py
│   └── repo_status.py
├── data/{aliases.yaml,schemas.yaml,tags.yaml}
├── candidates/
├── sources/{prs,docs,runs}/
├── wiki/{hardware,techniques,kernels,patterns,languages}/
├── artifacts/
├── queries/
└── references/
```

算子生成工作流的可安装 Skill 位于：

```text
integrations/cannbot-skills/ops/triton-kernelwiki-retriever/
├── SKILL.md
├── agents/openai.yaml
├── scripts/retrieve.py
└── references/integration-contract.md
```

- `sources/` 保存不可变来源摘要，不承担跨来源归纳。
- `wiki/` 保存可复用知识，每页通过 `sources` 引用一个或多个来源页。
- `artifacts/` 保存代码或实验制品；每个 bundle 必须有 `PROVENANCE.yaml`。
- `queries/` 是由 frontmatter 生成的导航索引，不参与额外打分。
- `candidates/` 保存来源候选与 include/defer/exclude 决策。

## 3. 安装

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

原生脚本只依赖 Python 和 PyYAML，可以直接从知识库根目录使用；包入口主要用于 Agent 集成和 JSON 输出。

## 4. KernelWiki 原生用法

查询页面：

```powershell
.\.venv\Scripts\python.exe kernelwiki\scripts\query.py `
  rmsnorm vecutil_low --architecture Ascend910B1 --compact

.\.venv\Scripts\python.exe kernelwiki\scripts\query.py `
  --type kernel --tag tail_mask --has-code --paths-only
```

读取页面、跟随来源并附带 artifact：

```powershell
.\.venv\Scripts\python.exe kernelwiki\scripts\get_page.py `
  kernel-rmsnorm-rope --follow-sources --include-code
```

全文和 artifact 检索：

```powershell
.\.venv\Scripts\python.exe kernelwiki\scripts\grep_wiki.py `
  pair_base --only all --context 2

.\.venv\Scripts\python.exe kernelwiki\scripts\grep_wiki.py `
  reference_rmsnorm_rope --only artifacts --files-only
```

校验、生成索引和统计：

```powershell
.\.venv\Scripts\python.exe kernelwiki\scripts\validate.py
.\.venv\Scripts\python.exe kernelwiki\scripts\generate-indices.py
.\.venv\Scripts\python.exe kernelwiki\scripts\repo_status.py
```

## 5. 查询分数

查询沿用 KernelWiki 的确定性文本相关性规则：标题命中 `+10`，标签字段命中 `+5`，正文每次命中 `+1` 且每个关键词最多 `+3`。通用别名通过 `data/aliases.yaml` 展开；架构组通过 `data/architecture-aliases.yaml` 展开为具体芯片型号；最终按 `(-score, path)` 稳定排序。

该分数只回答“页面文本与查询是否相关”，不代表证据质量、代码正确性或性能收益。证据强度由 `confidence`、`reproducibility`、`sources` 和 `performance_claims` 表达。

## 6. 给算子生成 Agent 的数据

Agent 的标准动作是：先 `query` 找页面，再 `get-page` 读取完整 Wiki 页，然后沿 `sources` 读取依据。包入口将这三步封装为 JSON：

```powershell
.\.venv\Scripts\python.exe -m triton_ascend_kb bundle `
  --root kernelwiki rmsnorm vecutil_low `
  --type kernel --architecture Ascend910B1 --json
```

```json
{
  "query": ["rmsnorm", "vecutil_low"],
  "retrieval": "kernelwiki_title_tag_body",
  "pages": [
    {
      "id": "kernel-rmsnorm-rope",
      "type": "wiki-kernel",
      "path": "wiki/kernels/rmsnorm-rope.md",
      "score": 21,
      "frontmatter": {"confidence": "experimental"},
      "body": "完整 Wiki 页面正文"
    }
  ],
  "source_excerpts": [
    {
      "id": "pr-triton-ascend-fixture-993",
      "path": "sources/prs/triton-ascend/PR-993.md",
      "excerpt": "来源页前 500 字符"
    }
  ],
  "warnings": []
}
```

代码调用：

```python
from pathlib import Path
from triton_ascend_kb import KernelWikiAgentAdapter, SearchFilters

bundle = KernelWikiAgentAdapter(Path("kernelwiki")).build(
    ["rmsnorm", "vecutil_low"],
    filters=SearchFilters(type="kernel", architecture="Ascend910B1"),
    limit=5,
)
agent_context = bundle.to_dict()
```

### 6.1 接入 cannBot `triton-op-generator`

将只读检索 Skill 链接到算子工程：

```bash
ln -s \
  /path/to/triton_ascend_kb_demo/integrations/cannbot-skills/ops/triton-kernelwiki-retriever \
  /path/to/operator-project/.claude/skills/triton-kernelwiki-retriever

export TRITON_ASCEND_KB_ROOT=/path/to/triton_ascend_kb_demo/kernelwiki
```

Phase 2 设计前调用：

```bash
python .claude/skills/triton-kernelwiki-retriever/scripts/retrieve.py \
  --mode design --workdir "$WORKDIR" --op-name LayerNorm \
  --semantic-op layer_norm --category normalization --kernel-type norm \
  --arch ascend910b1
```

输出写入 `$WORKDIR/knowledge/design_context.json`。`repair` 模式按 verifier 错误生成 `repair_iter_N.json`，`optimize` 模式生成 `optimize_iter_N.json`。三种模式保持官方文档、Wiki 案例、技术页和硬件页为独立检索通道，不进行跨通道重排，也不自动写回知识库。完整调用和消费契约见 Skill 的 `references/integration-contract.md`。

## 7. PR、源码和运行结果如何入库

真实 PR 的基础流程与 KernelWiki 一致：

1. 在 `candidates/*.yaml` 记录候选、纳入理由和决策。
2. 将 PR 元数据和证据摘要写入 `sources/prs/<repo>/PR-<N>.md`。
3. 将抓取或派生的代码放入 `artifacts/`，并记录来源 URL、许可证、模式和逐文件 SHA-256。
4. 将跨任务可复用的结论综合到 `wiki/kernels`、`wiki/techniques` 或 `wiki/patterns`。
5. Wiki 页用 `sources` 引用来源页，运行校验并重新生成索引。

现有源码按“来源页 + artifact”进入知识库：源码事实进入来源摘要，文件进入 artifact bundle；如果需要确认编译、正确性或 profiler 结论，再增加 `sources/runs/` 页面。不能复现或只能从代码推断的知识应标为 `inferred` 或 `experimental`，不能写成 `verified`。

仓库中的 PR-993、运行页和代码均标记为 `fixture: true`，只用于演示数据结构，不声称存在真实 PR 或真实数值加速。

## 8. Triton-Ascend 官方文档快照

当前知识库已从 [triton-lang/triton-ascend](https://github.com/triton-lang/triton-ascend) 的 `docs/en/**` 和 `docs/zh/**` 导入 249 个非空 Markdown/RST 页面，固定提交为：

```text
e4c692be02c38f3d5e99ccaf314bf4148fa84413
```

导入页位于 `kernelwiki/sources/docs/triton-ascend/`，快照清单位于 `kernelwiki/data/triton-ascend-doc-snapshot.yaml`。每页记录上游路径、固定提交、上游 URL、原始内容哈希、知识库正文哈希、语言、格式和 MIT 许可证。

刷新方式：

```powershell
git clone --depth 1 https://github.com/triton-lang/triton-ascend.git <checkout>

.\.venv\Scripts\python.exe kernelwiki\scripts\ingest_triton_ascend_docs.py `
  --repo-path <checkout> --retrieved-at 2026-07-15 --prune

.\.venv\Scripts\python.exe kernelwiki\scripts\validate.py
.\.venv\Scripts\python.exe kernelwiki\scripts\generate-indices.py
```

该脚本是确定性的：同一提交和同一抓取日期重复执行时 `changed=0`。`--prune` 只删除该脚本生成目录下已从上游消失的页面，不处理其他来源页。

查询示例：

```powershell
.\.venv\Scripts\python.exe kernelwiki\scripts\query.py `
  profiling msprof --repo triton-lang/triton-ascend --compact

.\.venv\Scripts\python.exe kernelwiki\scripts\query.py `
  tl.load mask --repo triton-lang/triton-ascend --limit 5
```

抓取的文档只进入 source 层。后续若要将“UB 溢出处理”“精度定位”或“Profiling 流程”提升为 Agent 可直接执行的技术知识，应新增经过审阅的 `wiki/techniques` 或 `wiki/patterns` 页面，并用 `sources` 引用这些官方文档页。

### 文档标签分类

文档分类不使用不透明模型，而采用 `data/doc-tagging-rules.yaml` 中可审计的路径和关键词规则：

- `symptoms`：错误或问题类型，例如 `compile_error`、`precision_mismatch`、`ub_overflow`、`illegal_memory_access`、`performance_bottleneck`。
- `kernel_types`：CANN 算子类别。
- `operators`：具体 API 或示例名，例如 `tl.load`、`topk`、`layer-norm`、`fused-attention`。
- `architectures`：仅使用 cannBot `npu-hardware-params.md` 产品映射表中的具体芯片型号，例如 `Ascend910B1`、`Ascend910_93`、`Ascend950PR`；DAV、A2/A3/A5、archXX 和产品族只作为一对多查询入口。
- `hardware_features`：存储层级、执行/搬运单元、数据通路和格式能力，例如 `unified_buffer`、`mte2`、`simt`、`l0c_to_ub`、`fp8`。

本版本不定义 `languages` 标签，不提供 `--language` 过滤，也不生成 `by-language.md`。

`kernel_types` 以以下三个 CANN 仓库的目录和 `classify_rule.yaml` 为依据，固定提交记录在 `data/operator-taxonomy-sources.yaml`：

| CANN 仓库 | kernel_types |
|---|---|
| `cann/ops-nn` | activation、control、conv、foreach、hash、index、loss、matmul、norm、optim、pooling、quant、rnn、vfusion |
| `cann/ops-math` | conversion、math、random |
| `cann/ops-transformer` | attention、ffn、gmm、mamba、mc2、mhc、moe、posembedding |

兼容词通过 `data/aliases.yaml` 归一化，例如 `elementwise → math`、`normalization → norm`、`gemm → matmul`、`rope → posembedding`。

当前分类结果：249 个导入页面中，32 页含 `symptoms`、141 页含 `kernel_types`、183 页含具体 `operators`；4 页具有精确芯片型号证据，形成 28 个型号关联；82 页形成 232 个硬件能力关联。重新应用规则不会改变文档正文，也不会破坏快照哈希：

```powershell
.\.venv\Scripts\python.exe kernelwiki\scripts\retag_triton_ascend_docs.py
.\.venv\Scripts\python.exe kernelwiki\scripts\validate.py
```

分类检索示例：

```powershell
# CANN attention 类算子文档
.\.venv\Scripts\python.exe kernelwiki\scripts\query.py `
  --repo triton-lang/triton-ascend --tag attention --compact

# 编译错误文档
.\.venv\Scripts\python.exe kernelwiki\scripts\query.py `
  --repo triton-lang/triton-ascend --symptom compile_error --compact

# UB 溢出文档
.\.venv\Scripts\python.exe kernelwiki\scripts\query.py `
  "UB overflow" --repo triton-lang/triton-ascend --symptom ub_overflow --compact
```

## 9. 数据源构建 Agent Demo

新增的 `triton-kb-source-builder` Skill 将文档、源码、PR、Issue 或 Agent Episode 转换为可审核的 KernelWiki 候选。Agent 负责阅读来源并填写 Source Packet；脚本按 `collect → build → verify → publish` 执行证据冻结、SHA-256、页面生成、受控标签检查、人工审批、全库校验和索引更新。

Skill 位于：

```text
integrations/cannbot-skills/ops/triton-kb-source-builder/
```

运行内置 PR fixture，仅生成候选：

```powershell
.\.venv\Scripts\python.exe `
  integrations\cannbot-skills\ops\triton-kb-source-builder\scripts\build.py `
  --root kernelwiki --workdir work\source-builder-demo `
  run --packet examples\source-builder\dynamic-tail-mask.yaml
```

输出位于 `work/source-builder-demo/dynamic-tail-mask-demo/`：

```text
source-packet.yaml
evidence-manifest.yaml
evidence/
candidates/source.md
candidates/wiki.md
tag-proposals.yaml
verification.yaml
```

内置示例产生两个证据快照，`tag_proposal_count=0`，校验结果为 `ready-for-review`。需要发布时，必须提供具名审核人：

```powershell
.\.venv\Scripts\python.exe `
  integrations\cannbot-skills\ops\triton-kb-source-builder\scripts\build.py `
  --root kernelwiki --workdir work\source-builder-demo `
  publish --build-id dynamic-tail-mask-demo --approved-by <reviewer>
```

未知受控标签只写入 `tag-proposals.yaml`，不会自动修改 `tags.yaml`。候选保持 `blocked`，直到维护者依据 `kernelwiki/data/tag-governance.yaml` 在独立变更中更新权威来源、词表和规则并完成回归测试。`architectures` 仍只使用 `npu-hardware-params.md` 的精确芯片型号；A2/A3/A5、DAV、archXX、SocVersion 和产品族仅作查询别名。

## 10. 测试与覆盖率

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_tests.ps1
```

脚本分别执行 UT、ST 和合并覆盖率，并强制最低门槛：UT 85%、ST 80%、合并 90%。实际结果写入 `reports/ut/`、`reports/st/` 和 `reports/combined/`；README 中的数字以最近一次完整运行结果为准。

本次实际结果如下，覆盖率统计范围为可安装实现 `src/triton_ascend_kb`；原生 `kernelwiki/scripts` 由 ST 通过真实子进程调用验证其接口契约。

| 测试集 | 用例 | 结果 | 综合覆盖率 | 语句覆盖率 | 分支覆盖率 |
|---|---:|---:|---:|---:|---:|
| UT | 47 | 47 passed | 88.06% | 90.45%（1099/1215） | 82.43%（427/518） |
| ST | 12 | 12 passed | 81.59% | 85.35%（1037/1215） | 72.78%（377/518） |
| UT + ST | 59 | 59 passed | 90.48% | 92.51%（1124/1215） | 85.71%（444/518） |
