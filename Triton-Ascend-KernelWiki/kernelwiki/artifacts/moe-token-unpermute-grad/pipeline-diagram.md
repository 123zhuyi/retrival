# moe_token_unpermute_grad - 流水编排分析

## 1. 核心公式（README L19-45）

正向（unpermute）：`O[i] = Σ_k probs[k] * permutedTokens[sortedIndices[k]]`

反向：
```
有 probs:
  probsGrad = Σ_{k}(unpermutedTokensGrad[i] * permutedTokens[sortedIndices[k]])   # 对 topK 求和
  permutedTokensGrad[sortedIndices[k]] = unpermutedTokensGrad[i] * probs[k]       # scatter 加权

无 probs (topK=1):
  permutedTokensGrad[sortedIndices[i]] = unpermutedOutputGrad[i]                  # 纯 scatter
```

## 2. 流水阶段

| 阶段 | ProbNone | ProbNotNone |
|---|---|---|
| 1. Gather | permutedTokens[sortedIndices]（probsGrad 不需要） | permutedTokens[sortedIndices] |
| 2. ProbsGrad | - | `Σ(grad * gathered)` over topK |
| 3. Scatter | `grad[sortedIndices] = unpermutedGrad` | `grad[sortedIndices] = unpermutedGrad * probs` |

## 3. 与 unpermute 配对

- 正向 `moe_token_unpermute`：gather + probs mul + reduce_sum
- 反向 `moe_token_unpermute_grad`：scatter 加权 + probsGrad reduce_sum

## 4. 阶段 -> Triton 抽象

全部可迁移（scatter + 元素乘 + reduce_sum，无 Cube）。AIV-only。