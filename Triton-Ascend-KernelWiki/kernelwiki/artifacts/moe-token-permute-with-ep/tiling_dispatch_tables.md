# moe_token_permute_with_ep - tiling 与 dispatch 分析

## A. 架构分发

`def.cpp` L66-67：仅 `ascend910b` + `ascend910_93`（A2/A3）。不支持 950/310P/Kirin。

## B. tiling key（8 个，与 token_permute 同构）

| key | 排序 | index_copy | padded_mode |
|---|---|---|---|
| 1 | OneCore | Normal | false |
| 2 | MultiCore | Normal | false |
| 3 | OneCore | SpiltD | false |
| 4 | MultiCore | SpiltD | false |
| 5 | OneCore | Normal | true |
| 6 | MultiCore | Normal | true |
| 7 | OneCore | SpiltD | true |
| 8 | MultiCore | SpiltD | true |

注：README L132 说仅支持 paddedMode=false（key 1-4 实际使用）。

## C. 核心计算（3 阶段，paddedMode=false）

公式（README L22-38）：
```
1. sortedIndicesFirst = argSort(indices)
2. sortedIndicesOut = argSort(sortedIndicesFirst)
3. 当 range[0] <= sortedIndices[i] < range[1]:
   permuteTokens[sortedIndices[i]-range[0]] = tokens[i//topK]
   permuteProbs[sortedIndices[i]-range[0]] = probs[i]
```

与 [[kernel-moe-token-permute]] 的差异：新增 probs copy + range 范围切片（EP 切分）。

## D. AIV-only

kernel 入口 L51-53：`if (g_coreType == AIC) return` - **纯 AIV，无 Cube**。

## E. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| argSort | `tl.sort` / 并行排序 | 排序得索引 |
| range 切片 | `if range[0] <= idx < range[1]` | EP 范围过滤 |
| index_copy tokens + probs | `tl.store` scatter copy（双输出） | tokens + probs 同时 copy |
| 8 tiling key | `if constexpr` | 排序×copy×padded |
| OneCore/MultiCore | `tl.program_id` + grid | 核数 |
| SpiltD | BLOCK_D constexpr | D 维分块 |