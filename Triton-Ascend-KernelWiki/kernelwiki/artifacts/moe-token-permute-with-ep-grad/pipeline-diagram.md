# MoeTokenPermuteWithEpGrad - Pipeline 抽象 (S6)

## 1. 计算流程

| 阶段 | 操作 | 单元 |
|------|------|------|
| 1 | range 过滤 sortedIndices | AIV |
| 2 | indexSelect gather permutedTokensOutputGrad | AIV |
| 3 | reshape (-1, topK, hiddenSize) | AIV |
| 4 | 跨 topK reduce sum -> token_grad | AIV |
| 5 | 若有 probs：gather + reduce -> probs_grad | AIV |

## 2. 阶段映射表

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移? |
|------|----------|------------|--------|
| range 过滤 | DataCopy + mask | `tl.where` mask | 是 |
| indexSelect | DataCopy 索引取数 | `tl.load(grad + idx * stride)` | 是 |
| reshape | view（无搬运） | view | 是 |
| reduce sum | Add 循环 | `tl.sum` | 是 |
| 写回 | DataCopy | `tl.store` | 是 |