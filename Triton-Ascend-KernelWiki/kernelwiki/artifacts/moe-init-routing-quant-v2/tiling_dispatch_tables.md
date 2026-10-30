# moe_init_routing_quant_v2 - tiling 与 dispatch 分析

## A. 架构分发

`def.cpp` L84-85：`ascend910b` + `ascend910_93`（arch32 kernel `moe_init_routing_quant_v2.cpp`）。L86-91：`ascend950`（arch35 kernel `moe_init_routing_quant_v2_apt.cpp`，opFile=..._apt）。README L7-12：支持 950/A3/A2。

## B. tiling key（3 档）

| key 范围 | 模式 | 说明 |
|---|---|---|
| 20000 | 静态 quant fullload | `MoeV2FullLoadQuant` 单阶段快速路径 |
| 21000 | 动态 quant fullload | `MoeV2FullLoadDynamicQuant` 单阶段 |
| 10000-11110 | 多阶段 | Sort -> ExpertTokenOut -> SrcToDst -> GatherQuant |

多阶段 tiling key 编码：
- 1xxxx：1core/multicore（0=onecore, 1=multicore，第 4 位）
- x0xxx/x1xxx：dropless(0)/drop(1)（dropPadMode，第 3 位）
- xx0xx/xx1xx：静态(0)/动态(1) quant（quantMode，第 2 位）

## C. AIV-only

arch32 kernel L43：`KERNEL_TYPE_MIX_AIV_1_0`（纯 AIV，1:0 配比）。arch35 kernel L43-45：`g_coreType == AIC return`。**全部 AIV-only，无 Cube。**

## D. 核心计算（6 步，README L25-83）

```
1. Sort: sortedExpertIdx, sortedRowIdx = keyValueSort(expertIdx)
2. SrcToDst: expandedRowIdx[sortedRowIdx[i]] = i
3. (dropless) expertTokensCountOrCumsum = Cumsum(Histogram(sortedExpertIdx))
4. (drop) expertTokensBeforeCapacity = Histogram(sortedExpertIdx)
5. Quant:
   - 静态: quantResult = round(x * scale + offset)
   - 动态: dynamicScale = row_max(abs(x[·scale])) / 127;  quantResult = round(x / dynamicScale)
6. Gather: expandedX[i] = quantResult[sortedRowIdx[i] % NUM_ROWS]
```

## E. V2 vs V1 差异

| 差异 | V1 (moe_init_routing_quant) | V2 |
|---|---|---|
| rowIdx 输入 | 有 | 删除 |
| drop/dropless | 无 | 有（dropPadMode） |
| 动态 quant | 无 | 有（quantMode=1） |
| expertTokens 统计 | 无 | 有（count/cumsum/beforeCapacity） |
| 架构 | A2/A3 | A2/A3 + 950(arch35) |
| 输出 dtype | INT8 | INT8/INT4 |
| 快速路径 | 无 | fullload 单阶段 |

## F. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| keyValueSort | `tl.sort` / 并行排序 | 排序 |
| expandedRowIdx 反向映射 | scatter store | 反向索引 |
| Histogram + Cumsum | `tl.histogram` + prefix sum | expert 统计 |
| 静态 quant | `tl.cast(round(x*scale+offset), int8/int4)` | 量化 |
| 动态 quant | `row_max(abs(x))/127` + `round(x/dyn_scale)` | 动态量化 |
| gather | `tl.load(quantResult + sorted_idx*stride)` | gather |
| drop/dropless 容量 | `if idx < expertCapacity` | drop 边界 |
| tiling key 分支 | `if constexpr` | 模式分支 |
| AIV-only | 单 program family | 无 Cube |