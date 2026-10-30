# moe_compute_expert_tokens - tiling 与 dispatch 分析

## A. 架构分发

`def.cpp` L44-48：`ascend910b` + `ascend910_93` + `ascend950` + `kirinx90` + `kirin9030`（全架构统一 config）。

## B. tiling key（3 个，按数据量）

| key | 模板 | 适用场景 |
|---|---|---|
| 1001 | `MoeComputeExpertTokensInt32SS` | 小数据量（single sort） |
| 1002 | `MoeComputeExpertTokensInt32M` | 中数据量 |
| 1003 | `MoeComputeExpertTokensInt32L` | 大数据量 |

## C. AIV-only

kernel 入口 L24：`KERNEL_TYPE_MIX_AIV_1_0` - **纯 AIV，无 Cube**。

## D. 核心计算（二分查找）

公式（README L21-23）：
```
out[i] = BinarySearch(sortedExperts, numExperts)
```

在已排序的 sortedExperts 数组中，对每个专家 i 二分查找其最后出现的行位置，输出累积 token 数。

## E. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| BinarySearch(sortedExperts, numExperts) | 自定义并行二分查找 / `tl.where` 累加 | 查找专家边界 |
| sortedExperts 已排序 | 前提条件 | 来自 init_routing 排序输出 |
| 3 tiling key 分支 | `if constexpr` | 数据量选择 |
| AIV-only | 单 program family | 无 Cube |