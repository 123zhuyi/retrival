# moe_gating_top_k_softmax - 流水编排分析

## 1. 核心公式（README L19-33）

```
1. softmaxOut = softmax(x, axis=-1)              # 行 softmax
2. yOut, expertIdxOut = topK(softmaxOut, k)      # 取 topk
3. rowIdxOut = arange(N*K).reshape(K,N).T        # 行索引
4. 若 finished[i]=True -> expertIdxOut = num_expert
```

## 2. 流水阶段

| 阶段 | 操作 | 说明 |
|---|---|---|
| 1. Softmax | `softmax(x, axis=-1)` | 行归一化 |
| 2. TopK | `topK(softmaxOut, k)` | 取 topk 值 + 索引 |
| 3. RowIdx | arange + reshape + transpose | 生成行索引 |
| 4. Finished 处理 | 条件填充 num_expert | finished 标记 |

## 3. 3 模板族（按数据量选择）

| 模板 | 适用场景 | 策略 |
|---|---|---|
| E_K_FullLoad | 小数据量 | expert×k 全装载到 UB，单次计算 |
| K_FullLoad | 中数据量 | k 全装载，expert 分块 |
| Perf | 大数据量 | 按 ColRange(expert 数) 优化：<8 / 8-64 / >64 |

## 4. 阶段 -> Triton 抽象

全部可迁移（softmax + topk + 索引生成，无 Cube）。AIV-only。3 模板族为数据量优化策略。