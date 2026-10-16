# base_api — 基础 API 检索（从 reference/ 找需要的 triton-ascend API）

## 场景
"实现某功能要用哪个 Triton API（tl.load/tl.store/tl.dot/tl.reduce 等）？签名/约束/平台支持在哪？" 从 `reference/` 找 API 卡。

## 路线
- 召回：`bm25`（别名默认开，中英 API 名/术语自动扩展），scope=`reference/`，`--kind api`。
- 重排：`bm25f`（默认，确定性）。
- 多跳补全：对选中的 API `neighbors` 取**配套约束卡 / 样例卡 / 相关接口**。

## 命令
```
python3 scripts/knowledge_query.py --knowledge-root <知识库根> recall \
  --method bm25 --query "<API名 同义/英文>" \
  --kind api --scope reference/ -k 8

# 选定后取整卡 + 多跳周边（约束/样例/相关接口）
python3 scripts/knowledge_query.py --knowledge-root <知识库根> get reference/<bundle>/<path>/<api>.md
python3 scripts/knowledge_query.py --knowledge-root <知识库根> neighbors reference/<bundle>/<path>/<api>.md --hops 1 -k 8
```

## 注入
`get` 整卡（签名 + 约束 + 出处链接）→ `neighbors` 把约束卡/样例卡/同主题接口一并带出，给一个连通的 API 知识面 → 引用 `resource` @sha。
