# moe_gating_top_k_softmax - tiling 与 dispatch 分析

## A. 架构分发

`def.cpp` L51-52：`ascend910b` + `ascend910_93`（arch32 kernel `moe_gating_top_k_softmax.cpp`）。L54-59：`ascend950`（arch35 kernel `moe_gating_top_k_softmax_apt.cpp`，opFile=..._apt）。README L7-12：支持 950/A3/A2。

## B. tiling key（18 个，3 模板族）

| key 范围 | 模板族 | dtype | 说明 |
|---|---|---|---|
| 0-5 | E_K_FullLoad | FP32/FP16/BF16 × BUFFER_NUM(1/2) | expert×k 全装载（小数据量） |
| 6-8 | K_FullLoad | FP32/FP16/BF16 × BUFFER_NUM(2) | k 全装载（中数据量） |
| 9-17 | Perf | FP32/FP16/BF16 × ColRange(SMALLER_THAN_8/FROM_8_TO_64/BIGGER_THAN_64) | 按列范围优化（大数据量） |

## C. AIV-only

kernel 入口 L53-55：`if (g_coreType == AIC) return` - **纯 AIV，无 Cube**。

## D. 核心计算

公式（README L19-33）：
```
1. softmaxOut = softmax(x, axis=-1)
2. yOut, expertIdxOut = topK(softmaxOut, k=k)
3. rowIdxOut = arange(N*K).reshape(K, N).transpose(1, 0)   # 行索引
4. 若 finished[i]=True，expertIdxOut 填 num_expert
```

## E. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| softmax(axis=-1) | `tl.softmax` / 在线 softmax | 行 softmax |
| topK | `tl.topk` / 自定义并行 topk | 取 topk |
| rowIdx reshape+transpose | `tl.arange` + reshape | 行索引生成 |
| finished 填 num_expert | `tl.where(finished, num_expert, idx)` | 条件填充 |
| 18 tiling key 分支 | `if constexpr` | 模板族×dtype |
| 3 模板族 | 数据量选择策略 | E_K/K 全装载 vs Perf 分块 |
| ColRange 优化 | 按 expert 数选策略 | <8 / 8-64 / >64 |