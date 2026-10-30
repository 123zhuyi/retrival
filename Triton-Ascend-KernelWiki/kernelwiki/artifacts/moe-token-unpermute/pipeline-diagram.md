# moe_token_unpermute - 流水编排分析

## 1. 核心公式（README L21-47）

```
有 probs:
  T[k] = permutedTokens[sortedIndices[k]]       # gather
  T[k] = T[k] * probs[k]                          # probs 乘
  O[i] = Σ_{k=i*topK}^{(i+1)*topK-1} T[k]        # topK 累加还原

无 probs (topK=1):
  O[i] = permutedTokens[sortedIndices[i]]         # 纯 gather
```

## 2. 流水阶段

| 阶段 | 操作 | 说明 |
|---|---|---|
| 1. Gather | `permutedTokens[sortedIndices[k]]` | 按 sortedIndices 离散 gather |
| 2. ProbsMul | `T[k] * probs[k]`（可选） | 加权 |
| 3. ReduceSum | `Σ over topK` | 还原原始 token 顺序 |

## 3. 与 token_permute 配对

- `moe_token_permute`：tokens -> permuteTokens（按 indices 排序展开）
- `moe_token_unpermute`：permutedTokens -> unpermutedTokens（按 sortedIndices gather + probs 加权 + topK 累加还原）

## 4. 阶段 -> Triton 抽象

全部可迁移（gather + 元素乘 + reduce_sum，无 Cube）。AIV-only（纯 elementwise）。