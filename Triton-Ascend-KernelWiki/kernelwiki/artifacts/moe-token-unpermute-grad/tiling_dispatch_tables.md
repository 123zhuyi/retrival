# moe_token_unpermute_grad - tiling 与 dispatch 分析

## A. 架构分发

`def.cpp` L76-77：仅 `ascend910b` + `ascend910_93`（A2/A3）。950 转调 `aclnnMoeFinalizeRoutingV2Grad`（README L121-129）。不支持 310P/Kirin。

## B. tiling key（4 个）

| key | padded_mode | haveProbs | 模板 |
|---|---|---|---|
| 0 | false | false | `MoeTokenUnpermuteGradProbNone` |
| 1 | false | true | `MoeTokenUnpermuteGradProbNotNone` |
| 10 | true | false | `MoeTokenUnpermuteGradProbNone` |
| 11 | true | true | `MoeTokenUnpermuteGradProbNotNone` |

## C. 核心计算（unpermute 反向）

公式（README L19-45）：
```
有 probs:
  # 正向: O[i] = Σ_k probs[k] * permutedTokens[sortedIndices[k]]
  # 反向:
  probsGrad = Σ_{k}(unpermutedTokensGrad * permutedTokens[sortedIndices])   # topK 求和
  permutedTokensGrad[sortedIndices[i]] = (unpermutedTokensGrad * probs)[i]  # scatter 加权

无 probs (topK=1):
  permutedTokensGrad[sortedIndices[i]] = unpermutedOutputGrad[i]            # 纯 scatter
```

## D. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| scatter permutedTokensGrad[sortedIndices] | `tl.store(grad + sorted_idx*stride, val)` | scatter 写回 |
| probsGrad topK 求和 | `tl.sum(grad * gathered, axis=topK)` | reduce_sum |
| 有 probs 加权 scatter | `grad * probs` 后 scatter | 元素乘 + scatter |
| 4 tiling key 分支 | `if constexpr` | padded×haveProbs |
| 2 模板 ProbNone/ProbNotNone | 独立 kernel / `if constexpr` | probs 有无 |