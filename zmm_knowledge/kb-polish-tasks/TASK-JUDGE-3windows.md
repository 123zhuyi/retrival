# 3 窗口并行：303 张 exp 卡 LLM 边判断（judge 批次）

## 背景

`okf_graph.py verify` 剩最后一条 FAIL：303 张 exp_* 卡从未经过 LLM 边判断。
候选集已生成并切好三份（含 focal 卡与 30 个候选的全部上下文，窗口无需再读库）：

| 窗口 | 输入文件 | focal 范围 | idx 范围 |
|---|---|---|---|
| J1 | `.build/judge/w1.json` | compute-pipeline exp 卡（148 张） | 0–147 |
| J2 | `.build/judge/w2.json` | memory-sram exp 卡（108 张） | 148–255 |
| J3 | `.build/judge/w3.json` | portability(33)+meta-skills(8)+algorithm-numerics(4)+autotuning(2)（47 张） | 256–302 |

知识库根：`KB=/home/w00836812/shixi_test02/zmm_knowledge/triton-ascend-kb.bak.20260807`
idx→卡 id 映射：`.build/judge/manifest.json`（已同步到 `.build/cand/manifest.json`）

## 每个窗口的任务（J1/J2/J3 相同，只换文件名与 idx 范围）

1. 读取自己窗口的输入文件 `$KB/.build/judge/wN.json`，结构为列表：
   `{idx, card: {id,title,description,tags,excerpt}, candidates: [{id,title,description,tags,excerpt} ×≤30]}`
2. 对每张 focal 卡，逐条判断候选是否值得建图谱边。**判定准则**：
   - 只保留与 focal 卡是**同一技术点**的候选：同一技术/同一根因/同一约束/同一修复链上下游。
     仅仅同目录、同维度、同算子类型**不构成**建边理由。
   - 每张 focal 保留 0–8 条，宁缺毋滥；没有合适的就是空列表。
   - `type` 取值：近重复技术用 `same_topic`；候选是 focal 规则的算子实例用 `exemplifies`；
     拿不准一律用 `related`。
   - `reason`：一句中文短句，点明共享的技术点（如"共享 int64 比较标量降级根因"）。
3. 产出 `$KB/.build/judge/workflow_output_wN.json`，**严格格式**：
   ```json
   {"result": {"results": [
     {"idx": 0, "kept": [{"target_id": "runbooks/xxx/exp_yyy.md", "type": "related", "reason": "…"}]},
     {"idx": 1, "kept": []}
   ]}}
   ```
   - 必须覆盖自己范围的**每一个 idx**（kept 可为空数组）
   - `target_id` 必须原样取自候选的 `id` 字段，不得杜撰
4. 上下文装不下就分批处理（如每批 10 张 focal，用 python/jq 切片），最后合并成一个输出文件。

## 纪律

- 只写 `$KB/.build/judge/workflow_output_wN.json` 这一个文件，不改库中任何其他文件
- 不 commit、不 push
- 不跑 okf_graph.py / knowledge_query.py / knowledge_lint.py（收尾统一跑）

## 收尾（三个窗口都完成后，任选其一或人工执行）

```bash
python3 $KB/.build/judge/finalize.py
```

finalize 做的事：合并三个输出 → 备份现有 edge_judgments.json → 跑 okf_judge_aggregate.py
→ 把旧判断并集合并回（aggregate 是全量覆盖语义）→ inject → viz → verify。
