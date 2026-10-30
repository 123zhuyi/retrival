# moe_token_permute - 流水编排分析

## 1. 核心公式（README L18-40，paddedMode=false）

```
1. sortedIndicesFirst = argSort(indices)
2. sortedIndicesOut = argSort(sortedIndicesFirst)
3. permuteTokens[sortedIndicesFirst[i]] = tokens[i // topK]
```

paddedMode=true：`permuteTokens[i] = tokens[indices[i]]` + `sortedIndicesOut = indices`。

## 2. 3 阶段流水

| 阶段 | 类 | 操作 |
|---|---|---|
| 1. Sort1 | `MoeSortOneCore`/`MultiCore` | argSort(indices) -> sortedIndicesFirst |
| 2. Sort2 | `MoeSortOneCore`/`MultiCore` | argSort(sortedIndicesFirst) -> sortedIndicesOut |
| 3. IndexCopy | `MoeindexCopyOp`/`MoeindexCopySpiltDOp` | permuteTokens[sortedIndicesFirst[i]] = tokens[i//topK] |

## 3. 阶段 -> Triton 抽象

全部可迁移（排序 + scatter copy，无 Cube）。AIV-only。SpiltD 为 D 维切分优化变体。