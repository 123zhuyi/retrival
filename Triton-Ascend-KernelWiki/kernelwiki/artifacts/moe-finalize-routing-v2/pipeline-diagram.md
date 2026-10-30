# MoeFinalizeRoutingV2 — Pipeline 抽象 (S6)

## 1. 计算流程

| 阶段 | 操作 | 单元 |
|------|------|------|
| 1 | 加载 expandedRowIdx 索引 | AIV |
| 2 | Gather expandedX 按索引取行 | AIV |
| 3 | 加载 bias 按 expertIdx 取偏置 | AIV |
| 4 | expandedX_gathered + bias | AIV |
| 5 | * scales 加权 | AIV |
| 6 | 跨 K 累加 | AIV |
| 7 | + x1 + x2Optional | AIV |
| 8 | 写回 out | AIV |

## 2. 阶段映射表

| 阶段 | CANN 机制 | 目的 | Triton 抽象 | 可迁移? |
|------|----------|------|------------|--------|
| 数据读入 | DataCopy GM->UB | 载入 expandedX/idx/scales/bias | `tl.load` | 是 |
| Gather | 索引取数 | 按 expandedRowIdx 取行 | `tl.load(ptr + idx[j] * stride)` | 是 |
| 加权 | Muls | `* scales` | `* scales` | 是 |
| 加偏置 | Add | `+ bias[expertid]` | `+ tl.load(bias + expertid)` | 是 |
| 跨 K 累加 | Add 循环 | `Σ_k` | `tl.sum` 或循环 | 是 |
| 加 x1/x2 | Add | `+ x1 + x2Optional` | `+ x1 + x2` | 是 |
| 写回 | DataCopy | 输出 out | `tl.store` | 是 |