# moe_token_permute - tiling 与 dispatch 分析

## A. 架构分发

`def.cpp` L51-52：仅 `ascend910b` + `ascend910_93`（A2/A3）。950 在框架层转调 `aclnnMoeInitRoutingV2`（README L105-112），不走本 kernel。

## B. tiling key（8 个）

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

注：README L79 说目前仅支持 paddedMode=false（key 1-4 实际使用）。

## C. 核心计算（3 阶段）

公式（README L18-40，paddedMode=false）：
```
1. sortedIndicesFirst = argSort(indices)
2. sortedIndicesOut = argSort(sortedIndicesFirst)
3. permuteTokens[sortedIndicesFirst[i]] = tokens[i // topK]
```

paddedMode=true 时简化为 `permuteTokens[i] = tokens[indices[i]]` + `sortedIndicesOut = indices`。

## D. AIV-only

kernel 入口 L47-49：`if (g_coreType == AIC) return` - **纯 AIV 执行，无 Cube**。

## E. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| argSort(indices) | `tl.sort` / 并行排序 | 排序得索引 |
| argSort(sortedIndicesFirst) | `tl.sort` | 二次排序 |
| index_copy permuteTokens[idx]=tokens[i//topK] | `tl.store(permute + idx*stride, tl.load(tokens + (i//topK)*stride))` | scatter copy |
| 8 tiling key 分支 | `if constexpr` | 排序×copy×padded |
| OneCore/MultiCore | `tl.program_id` + grid | 核数 |
| SpiltD（D 维切分） | BLOCK_D constexpr | D 维分块 |