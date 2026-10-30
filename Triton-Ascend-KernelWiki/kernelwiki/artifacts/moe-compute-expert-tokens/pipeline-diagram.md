# moe_compute_expert_tokens - 流水编排分析

## 1. 核心公式（README L21-23）

```
out[i] = BinarySearch(sortedExperts, numExperts)
```

在已排序的 sortedExperts 中，对每个专家 i（0..numExperts-1）二分查找其最后出现的行位置。

## 2. 流水阶段

| 阶段 | 操作 | 说明 |
|---|---|---|
| 1. 遍历专家 | for i in 0..numExperts | 每个专家 |
| 2. 二分查找 | BinarySearch(sortedExperts, i) | 找专家 i 的边界行 |
| 3. 输出 | out[i] = 累积行数 | 累积 token 数 |

## 3. 3 模板（按数据量）

| 模板 | 场景 | 策略 |
|---|---|---|
| SS | 小数据量 | 单核全装载 |
| M | 中数据量 | 分块 |
| L | 大数据量 | 多核并行 |

## 4. 阶段 -> Triton 抽象

全部可迁移（二分查找，无 Cube）。AIV-only。Triton 中需自定义并行二分查找或用 `tl.where` 累加实现。