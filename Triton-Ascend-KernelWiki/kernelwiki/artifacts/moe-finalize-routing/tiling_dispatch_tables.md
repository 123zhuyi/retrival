# moe_finalize_routing - tiling 与 dispatch 分析

## A. 架构分发

`def.cpp` L79-80：`ascend910b` + `ascend910_93`（A2/A3）。L82-84：`kirinx90` + `kirin9030`（仅 FP16/FLOAT，无 BF16）。不支持 950/310P。

## B. tiling key（20000-20018）

19 个 tiling key，编码 dtype × 模板变体：

| key | dtype | 模板变体 | 说明 |
|---|---|---|---|
| 20000 | FLOAT | CutK | FP32 CutK |
| 20001 | FP16 | CutK | FP16 CutK |
| 20002 | BF16 | CutK | BF16 CutK（310P 排除） |
| 20003 | BF16 | Base | BF16 基准 |
| 20004 | FLOAT | Db | FP32 DoubleBuffer |
| 20005 | FP16 | Db | FP16 DoubleBuffer |
| 20006 | FLOAT | CuthK2 | FP32 Cuth K2 |
| 20007 | FP16 | CuthK2 | FP16 Cuth K2 |
| 20008 | BF16 | CuthK2 | BF16 Cuth K2 |
| 20009 | FLOAT | CuthK4 | FP32 Cuth K4 |
| 20010 | FP16 | CuthK4 | FP16 Cuth K4 |
| 20011 | BF16 | CuthK4 | BF16 Cuth K4 |
| 20012 | FLOAT | Cuth | FP32 Cuth |
| 20013 | FP16 | Cuth | FP16 Cuth |
| 20014 | BF16 | Cuth | BF16 Cuth |
| 20015 | FLOAT | DbAllBias | FP32 Db + all bias |
| 20016 | FP16 | DbAllBias | FP16 Db + all bias |
| 20017 | BF16 | AllBias | BF16 all bias |
| 20018 | FLOAT | CuthK2Optimized | FP32 CuthK2 Optimized |

## C. 核心计算（gather + scale + add）

公式（README L21-27）：
```
expertId = expandedExpertIdx[i, k]
out[i, j] = x1[i, j] + x2Optional[i, j]
            + Σ_{k=0}^{K}(scales[i, k] * (expandedX[expandedRowIdx[i+k*numRows], j]
                                           + bias[expertId, j]))
```

Triton 等价：`tl.load(expanded_x + row_idx * stride) * scale + bias` 累加 + `x1 + x2`。

## D. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| expandedRowIdx 索引 gather | `tl.load(expanded_x + row_idx*stride)` | 离散 gather |
| scales 加权求和 | `sum += tl.load(expanded_x) * scale` | 元素乘加 |
| bias[expertId] | `tl.load(bias + expert_id*stride)` | 专家偏置 |
| 19 tiling key 分支 | `if constexpr` / launcher | dtype×变体 |
| BUFFER_NUM=1 | 单 buffer（编译器调度） | CANN-specific |
| PARALLEL_NUM=2 | 并行因子 | CANN-specific（优化参数） |