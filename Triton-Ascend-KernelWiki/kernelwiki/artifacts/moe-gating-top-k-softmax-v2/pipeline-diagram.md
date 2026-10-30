# MoeGatingTopKSoftmaxV2 - Pipeline 抽象 (S6)

## 1. 计算流程

### renorm=0（先 softmax 后 topk）
| 阶段 | 操作 | 单元 |
|------|------|------|
| 1 | softmax(x, axis=-1) | AIV |
| 2 | topk(softmaxResult, k) -> yOut, expertIdxOut | AIV |
| 3 | finished 标记替换 | AIV |

### renorm=1（先 topk 后 softmax）
| 阶段 | 操作 | 单元 |
|------|------|------|
| 1 | topk(x, k) -> topkOut, expertIdxOut | AIV |
| 2 | softmax(topkOut, axis=-1) -> yOut | AIV |
| 3 | finished 标记替换 | AIV |

## 2. 阶段映射表

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移? |
|------|----------|------------|--------|
| softmax | SoftMax API | `tl.softmax` | 是 |
| topk | TopK API | `tl.topk` | 是 |
| finished 替换 | Select | `tl.where` | 是 |
| 写回 | DataCopy | `tl.store` | 是 |