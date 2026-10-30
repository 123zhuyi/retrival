# moe_init_routing_quant - 流水编排分析

## 1. 核心公式（README L19-29）

```
1. Sort: expandedExpertIdx, sortedRowIdx = keyValueSort(expertIdx, rowIdx)
2. SrcToDst: expandedRowIdx[sortedRowIdx[i]] = i
3. Gather+Quant: expandedX[i] = quant(x[sortedRowIdx[i] % numRows], scale, offset)
```

## 2. 3 阶段流水

| 阶段 | 类 | 操作 | 说明 |
|---|---|---|---|
| 1. Sort | `MoeSortOneCore` / `MoeSortMultiCore` | 按 expertIdx 排序得 expandedExpertIdx + sortedRowIdx | 归并排序 |
| 2. SrcToDst | `MoeSrcToDstOp` | 反向映射 expandedRowIdx[sortedRowIdx[i]]=i | scatter |
| 3. GatherOut+Quant | `MoeGatherOut` / `MoeGatherOutSmallActiveRow` | gather x[sortedRowIdx] 并量化为 INT8 | gather + cast |

## 3. 阶段 -> Triton 抽象

全部可迁移（排序 + scatter + gather + quant，无 Cube）。AIV-only。

## 4. 量化公式

`expandedX = round(x * scale + offset)`，cast to INT8。