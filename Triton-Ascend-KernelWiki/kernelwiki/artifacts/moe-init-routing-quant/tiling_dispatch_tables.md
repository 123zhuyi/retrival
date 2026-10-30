# moe_init_routing_quant - tiling 与 dispatch 分析

## A. 架构分发

`def.cpp` L59-60：仅 `ascend910b` + `ascend910_93`（A2/A3）。不支持 950/310P/Kirin/A3（README L7-9：A3 也不支持）。

## B. tiling key（4 个）

| key | 排序 | gather | 说明 |
|---|---|---|---|
| 1 | OneCore | Normal | 单核排序 + 标准 gather |
| 2 | MultiCore | Normal | 多核排序 + 标准 gather |
| 3 | OneCore | SmallActiveRow | 单核 + 小 active row 变体 |
| 4 | MultiCore | SmallActiveRow | 多核 + 小 active row 变体 |

## C. 核心计算（3 阶段）

公式（README L19-29）：
```
1. Sort: expandedExpertIdx, sortedRowIdx = keyValueSort(expertIdx, rowIdx)
2. SrcToDst: expandedRowIdx[sortedRowIdx[i]] = i
3. Gather+Quant: expandedX[i] = quant(x[sortedRowIdx[i] % numRows])
```

## D. AIV-only

kernel 入口 L36-38：`if (g_coreType == AIC) return` - **纯 AIV 执行，无 Cube 参与**。

## E. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| keyValueSort | `tl.sort` / 自定义并行排序 | 排序 |
| expandedRowIdx 反向映射 | scatter store `tl.store(idx*stride, i)` | 反向索引 |
| gather x[sortedRowIdx] | `tl.load(x + sorted_idx*stride)` | 离散 gather |
| quant to INT8 | `tl.cast(round(x*scale+offset), tl.int8)` | 量化 |
| 4 tiling key 分支 | `if constexpr` | 排序×gather 变体 |
| OneCore/MultiCore | `tl.program_id` + grid | 核数分支 |