# moe_init_routing_quant_v2 - 流水编排分析

## 1. 核心计算（6 步，README L25-83）

```
1. Sort: sortedExpertIdx, sortedRowIdx = keyValueSort(expertIdx)
2. SrcToDst: expandedRowIdx[sortedRowIdx[i]] = i
3. (dropless) expertTokensCumsum = Cumsum(Histogram(sortedExpertIdx))
   (drop)     expertTokensBeforeCapacity = Histogram(sortedExpertIdx)
4. Quant:
   静态: quantResult = round(x * scale + offset)
   动态: dynScale = row_max(abs(x [· scale])) / 127;  quantResult = round(x / dynScale)
5. Gather: expandedX[i] = quantResult[sortedRowIdx[i] % NUM_ROWS]
6. (drop) 容量截断/补零
```

## 2. 多阶段流水

| 阶段 | 类 | 操作 |
|---|---|---|
| 1. Sort | `MoeV2SortOneCore`/`MultiCore` | 按 expertIdx 排序 |
| 2. ExpertTokenOut | `MoeV2ExpertTokenOut`(/RegBase/Simt) | 直方图+前缀和 |
| 3. SrcToDst | `MoeV2SrcToDstOp`/`WithCapacity`/`AndGather` | 反向映射（drop 时带容量） |
| 4. GatherQuant | `MoeV2GatherQuant`/`GatherDynamicQuant`(/Dropless/Droppad) | gather + 量化 |

## 3. fullload 快速路径

tiling key 20000/21000：`MoeV2FullLoadQuant`/`MoeV2FullLoadDynamicQuant` 单阶段融合全部步骤（小数据量场景）。

## 4. 阶段 -> Triton 抽象

全部可迁移（排序+scatter+histogram+quant+gather，无 Cube）。arch35 regbase/simt 机制为 CANN 专有。