# MoeInitRoutingV3 - Pipeline 抽象 (S6)

## 1. 计算流程

| 阶段 | 入口 | 单元 | 操作 |
|------|------|------|------|
| 1 Sort | mrgsort / sort_one_core / sort_multi_core | AIV | keyValueSort(expertIdx) -> sortedExpertIdx/sortedRowIdx |
| 2 Map | row_idx_gather | AIV | expandedRowIdxOut[sortedRowIdx[i]]=i |
| 3 Histogram | expert_tokens_count | AIV | drop 模式直方图统计 |
| 4 Quant | gather_dynamic_quant / gather_static_quant | AIV | 动态/静态量化 -> quantResult + expanded_scale |
| 5 Gather | gather_out | AIV | expandedXOut[i]=quantResult[sortedRowIdx[i]%NUM_ROWS] |
| 6 WriteBack | - | AIV | 写回 4 个输出 |

## 2. 阶段映射表

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移? |
|------|----------|------------|--------|
| 排序 | MrgSort 归并排序 | `tl.sort` 或归并 | 是 |
| 位置映射 | DataCopy scatter | `tl.store(out + idx)` | 是 |
| 直方图 | Add 累加 | `tl.atomic_add` bincount | 是 |
| 动态量化 | Muls + Max + Round | `tl.max/tl.round` | 是 |
| 静态量化 | Muls + Add | `x*scale+offset` | 是 |
| Gather | 索引取数 | `tl.load(ptr + idx)` | 是 |
| 写回 | DataCopy | `tl.store` | 是 |