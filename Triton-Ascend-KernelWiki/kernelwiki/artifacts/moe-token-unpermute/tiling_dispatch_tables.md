# moe_token_unpermute - tiling 与 dispatch 分析

## A. 架构分发

`def.cpp` L75-76：`ascend910b` + `ascend910_93`（A2/A3，BF16/FP16/FP32）。L78-118：`ascend310p` + `kirinx90` + `kirin9030`（仅 FP16/FP32）。README L7-14：支持 950（但 L109-115 说 950 转调 `aclnnMoeFinalizeRoutingV2`）。

## B. tiling key（~16 个）

kernel 入口（`moe_token_unpermute.cpp` L33-87），编码 dtype × haveProbs × mix mode（probs dtype 与 token dtype 不同）：

| key | token dtype | probs dtype | haveProbs | mix |
|---|---|---|---|---|
| 0 | BF16 | BF16 | false | - |
| 1 | BF16 | BF16 | true | - |
| 2 | FP16 | FP16 | false | - |
| 3 | FP16 | FP16 | true | - |
| 4 | FP32 | FP32 | false | - |
| 5 | FP32 | FP32 | true | - |
| 11 | FP16 | BF16 | true | mix |
| 13 | FP32 | BF16 | true | mix |
| 17 | BF16 | FP16 | true | mix |
| 21 | FP32 | FP16 | true | mix |
| 25 | BF16 | FP32 | true | mix |
| 27 | FP16 | FP32 | true | mix |

310P/Kirin（`__CCE_AICORE__==200` 分支）仅 FP16/FP32。

## C. 核心计算（gather + probs mul + reduce_sum）

公式（README L21-47）：
```
有 probs:
  T[k] = permutedTokens[sortedIndices[k]]
  T[k] = T[k] * probs[i][j]      # i=k//topK, j=k%topK
  O[i] = Σ_{k=i*topK}^{(i+1)*topK-1} T[k]

无 probs (topK=1):
  T[i] = permutedTokens[sortedIndices[i]]
  O[i] = T[i]
```

## D. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| gather permutedTokens[sortedIndices] | `tl.load(permuted + sorted_idx*stride)` | 离散 gather |
| probs 乘 | `tl.load(probs) * gathered` | 元素乘 |
| topK 累加求和 | `tl.sum(gathered, axis=topK)` | reduce_sum |
| ~16 tiling key 分支 | `if constexpr` | dtype×probs 分支 |
| 950 转调 finalize_routing_v2 | README L109-115 | 框架层路由 |