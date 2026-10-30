# MoeInitRoutingV2Grad - Pipeline 抽象 (S6)

## 1. 计算流程

| 阶段 | 操作 | 单元 |
|------|------|------|
| 1 | 加载 expandedRowIdx[i*topK+t] | AIV |
| 2 | Gather gradExpandedX 按索引取行 | AIV |
| 3 | 跨 t 累加（reduce sum over topK） | AIV |
| 4 | 写回 gradX | AIV |

## 2. 阶段映射表

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移? |
|------|----------|------------|--------|
| 索引加载 | DataCopy | `tl.load(expandedRowIdx)` | 是 |
| Gather | 索引取数 | `tl.load(gradExpandedX + idx * stride)` | 是 |
| 跨 K 累加 | Add 循环 / 二路归并 | `tl.sum` 或循环 | 是 |
| 写回 | DataCopy | `tl.store` | 是 |