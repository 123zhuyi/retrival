# moe_token_permute_with_ep - 流水编排分析

## 1. 核心公式（README L22-38，paddedMode=false）

```
1. sortedIndicesFirst = argSort(indices)
2. sortedIndicesOut = argSort(sortedIndicesFirst)
3. 当 range[0] <= sortedIndices[i] < range[1]:
   permuteTokens[sortedIndices[i]-range[0]] = tokens[i//topK]
   permuteProbs[sortedIndices[i]-range[0]] = probs[i]
```

paddedMode=true：`permuteTokens[i]=tokens[indices[i]]` + `sortedIndicesOut=indices`。

## 2. 3 阶段流水

| 阶段 | 类 | 操作 |
|---|---|---|
| 1. Sort1 | `MoeSortOneCore`/`MultiCore` | argSort(indices) -> sortedIndicesFirst |
| 2. Sort2 | `MoeSortOneCore`/`MultiCore` | argSort(sortedIndicesFirst) -> sortedIndicesOut |
| 3. IndexCopy | `MoeindexCopyOp`/`MoeindexCopySpiltDOp` | tokens + probs 双输出，按 range 切片 |

## 3. 与 token_permute 差异

| 差异 | token_permute | token_permute_with_ep |
|---|---|---|
| probs 输入 | 无 | 有（OPTIONAL） |
| permute_probs 输出 | 无 | 有 |
| range 属性 | 无 | 有（EP 切分范围） |
| index_copy | 单 tokens | tokens + probs 双输出 |
| 切片逻辑 | numOutTokens | range[0]<=idx<range[1] |
| 回退 | - | range 空时回退到 token_permute（README L133） |

## 4. 阶段 -> Triton 抽象

全部可迁移（排序 + 条件 scatter copy，无 Cube）。AIV-only。