# 窗口 D 任务书：合并挂载 review + 索引/图谱/lint 收尾

## 0. 背景（必读）

Triton-Ascend 知识库从 raw 经验库批量迁移 303 条 YAML 卡片时，有 ~170 条被**合并挂载**到 20 个已有 runbook 卡片上（在「已知实例」区域追加「### 新增已知实例（from raw experience）」反链块）。

合并用的是 concept 关键词匹配，**存在主题漂移风险**：部分目标卡被挂了 9~19 条经验，其中可能混入了主题不符的挂载。

你负责两件事：
1. **Review 并修正合并挂载质量**
2. **收尾**：等窗口 A/B/C 打磨完成后，更新索引、重建图谱、跑 lint

## 1. 工作范围

目标库：`/home/w00836812/shixi_test02/zmm_knowledge/triton-ascend-kb.bak.20260807/`

**你只做：**
- 已有 runbook（非 exp_ 开头的 .md）上挂载的反链块 review
- 各分类 `index.md`、根 `index.md`、图谱、检索索引、lint

**不要做：**
- 不碰 exp_*.md 正文打磨（窗口 A/B/C 的活）
- 不与 A/B/C 同时改同一文件——你的挂载 review 只动「### 新增已知实例」块，不碰卡片正文

## 2. 任务 1：合并挂载 review

### 2.1 找出热点目标卡

```bash
cd /home/w00836812/shixi_test02/zmm_knowledge/triton-ascend-kb.bak.20260807
for f in runbooks/*/*.md; do
  n=$(grep -c '^- `exp_' "$f" 2>/dev/null || echo 0)
  [ "$n" -gt 3 ] && echo "$n  $f"
done | sort -rn
```

已知挂载最重的卡（挂载数，来自批量报告）：
- `compute-pipeline/scalar-load-channel-index.md`（~19）
- `compute-pipeline/pad-slot-early-exit.md`（~18）
- `memory-sram/overlapping-t-tiling.md`（~18）
- `memory-sram/vectorized-coalesced-io.md`（~17）
- `memory-sram/block-ptr-vs-scalar-load.md`（~17）
- `compute-pipeline/cube-core-num-fixed.md`（~12）
- `compute-pipeline/core-mapping-strategy-selection.md`（~12）

### 2.2 逐条 review 挂载

对每张热点卡，读取卡片主题，再逐条核对挂载的 exp 条目：

- 原始 YAML 在 `/home/w00836812/shixi_test02/cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/raw/experiences/`
- **主题一致** → 保留，并可把一句话备注补得更具体
- **主题不符** → 从该卡移除这条反链块；该 exp 在对应分类下已有自己的 exp_*.md 卡片（303 条全部已建卡），所以移除反链不丢内容
- 拿不准的**保留**并在行尾加 `（待复核）`

判定标准：挂载条目的优化技术必须与宿主卡的 OPT 原则是**同一技术点**，不是仅仅同分类/同维度。例如 `block-ptr-vs-scalar-load.md` 只应收「block_ptr boundary_check 标量开销 / 手动指针替代」相关经验，不应收一般的 attention tiling 经验。

### 2.3 confidence 校准

```bash
grep -l '^confidence: provisional' runbooks/*/exp_*.md
```

对 provisional 卡片：有 code_snippet 路径（`raw_experiences/extracted_kernels/`）且性能收益具体的 → 可升 `verified`；纯经验描述无出处的 → 保持 `provisional`。

## 3. 任务 2：收尾（等 A/B/C 完成后执行）

### 3.1 检查索引完整性

```bash
# 每个 exp_*.md 都应出现在对应分类 index.md 中
for d in memory-sram compute-pipeline portability algorithm-numerics autotuning meta-skills; do
  echo "== $d =="
  for f in runbooks/$d/exp_*.md; do
    b=$(basename "$f")
    grep -q "$b" runbooks/$d/index.md || echo "MISSING: $b"
  done
done
```

缺失的补进 index.md（格式参照已有条目）。

### 3.2 重建图谱与检索索引

```bash
KB=/home/w00836812/shixi_test02/zmm_knowledge/triton-ascend-kb.bak.20260807
SCRIPTS=/home/w00836812/shixi_test02/zmm_knowledge/Triton-Ascend-KernelWiki/skills

python3 $SCRIPTS/triton-knowledge-ingest/scripts/okf_graph.py --knowledge-root $KB candidates
python3 $SCRIPTS/triton-knowledge-ingest/scripts/okf_graph.py --knowledge-root $KB inject
python3 $SCRIPTS/triton-knowledge-ingest/scripts/okf_graph.py --knowledge-root $KB viz
python3 $SCRIPTS/triton-knowledge-ingest/scripts/okf_graph.py --knowledge-root $KB verify
python3 $SCRIPTS/triton-knowledge-query/scripts/knowledge_query.py --knowledge-root $KB build
```

### 3.3 跑 lint

```bash
python3 $SCRIPTS/triton-knowledge-lint/scripts/knowledge_lint.py --knowledge-root $KB
```

逐项修复 blocker；warn 记录并汇总。

## 4. 验收标准

- [ ] 热点目标卡挂载逐条过一遍，不符的已移除
- [ ] 所有 exp_*.md 出现在分类 index.md
- [ ] `okf_graph.py verify` 通过
- [ ] `knowledge_lint.py` 0 blocker
- [ ] 输出最终改动统计（新增/修改文件数、移除的反链数）

## 5. 冲突回避

- 挂载 review 只动「### 新增已知实例（from raw experience）」块及其周边
- 收尾步骤必须在 A/B/C 完成后进行，避免图谱索引基于半成品重建
- 不 commit、不 push（由用户统一处理）
