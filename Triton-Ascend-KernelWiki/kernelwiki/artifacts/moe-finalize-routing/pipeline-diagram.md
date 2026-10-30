# moe_finalize_routing - 流水编排分析

## 1. 核心公式（README L21-27）

```
out[i, j] = x1[i, j] + x2Optional[i, j]
            + Σ_{k=0}^{K}(scales[i, k]
              * (expandedX[expandedRowIdx[i+k*numRows], j]
                 + bias[expandedExpertIdx[i, k], j]))
```

## 2. 流水阶段

| 阶段 | 操作 | 说明 |
|---|---|---|
| 1. 索引 gather | 读 expandedRowIdx / expandedExpertIdx | INT32 gather |
| 2. expandedX gather | 按索引读 expandedX + bias | 离散 load |
| 3. 加权求和 | scales × (expandedX + bias) | 元素乘加 |
| 4. 累加 | Σ over K experts + x1 + x2 | 归约 |
| 5. 写回 | out[i, j] | store |

## 3. 阶段 -> Triton 抽象

全部可迁移（这是纯 elementwise gather+scale+add，无 Cube/AIC 参与）。`BUFFER_NUM=1`、`PARALLEL_NUM=2` 为 CANN 优化参数。