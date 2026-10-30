# MoeTokenUnpermuteWithEp - Pipeline 抽象 (S6)

## 1. 计算流程

| 阶段 | 操作 | 单元 |
|------|------|------|
| 1 | range 过滤 sortedIndices | AIV |
| 2 | indexSelect gather permutedTokens | AIV |
| 3 | 若有 probs：乘 probs_{i,j} | AIV |
| 4 | 跨 topK reduce sum -> unpermuted_tokens | AIV |

## 2. 阶段映射表

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移? |
|------|----------|------------|--------|
| range 过滤 | DataCopy + mask | `tl.where` mask | 是 |
| indexSelect | DataCopy 索引取数 | `tl.load(ptr + idx * stride)` | 是 |
| 乘 probs | Muls | `* probs` | 是 |
| reduce sum | Add 循环 | `tl.sum` | 是 |
| 写回 | DataCopy | `tl.store` | 是 |