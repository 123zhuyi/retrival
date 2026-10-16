---
title: Softmax / LogSoftmax 算子优化
category: ops
tags: [softmax, logsoftmax, online-softmax, reduction, numerical-stability]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# Softmax / LogSoftmax 算子优化

## 概述

Softmax 将实值向量归一化为概率分布：`softmax(x)_i = exp(x_i) / Σ exp(x_j)`。  
LogSoftmax 输出对数概率：`log_softmax(x)_i = x_i - log(Σ exp(x_j))`，数值更稳定。

在 Triton-Ascend 中，Softmax 的核心挑战是：
1. **两趟访存开销**：标准实现需两次遍历（先求 max，再求 exp+sum）
2. **数值溢出**：fp16/bf16 下 `exp(x)` 容易溢出，需减去最大值稳定化
3. **归约通信**：跨 tile 归约需要同步，影响并行度

## 检测标准

满足以下条件时归类为 Softmax：
- PyTorch 原语：`F.softmax(x, dim=...)` 或 `F.log_softmax(x, dim=...)`
- 归约沿指定维度（inner dim 或 outer dim），输出形状与输入相同
- 不含跨 batch 归约（跨 batch 属于 Reduction）

## 适用条件

| 条件 | 说明 |
|------|------|
| 数据类型 | fp16、bf16、fp32；fp8 需特殊 UB 分片处理 |
| inner dim N | N≥64 时向量化收益显著；N 超过 UB 容量需分块 |
| outer dim M | M≥1，通常每个 program 处理一行 |

## 经验路由

| 场景 | 经验ID | 提升% | 关键技术 | 经验文件 |
|------|--------|-------|----------|----------|
| 标准 Softmax（inner dim） | exp_softmax_inner_002 | 30% | 在线 max+sum 单趟，数值稳定 | [yaml](../../raw/experiences/softmax/exp_softmax_inner_002.yaml) |
| 非 inner dim Softmax | exp_softmax_non_inner_001 | 25% | 外层维度归约，转置后处理 | [yaml](../../raw/experiences/softmax/exp_softmax_non_inner_001.yaml) |
| Online Softmax（Attention 内融合） | exp_softmax_online_001 | 50% | 运行 m_i/l_i 迭代更新，消除完整 NxN 矩阵 | [yaml](../../raw/experiences/softmax/exp_softmax_online_001.yaml) |
| LogSoftmax 前向 | exp_log_softmax_001 | 30% | 两趟：max+sum → log-sum-exp 单核 | [yaml](../../raw/experiences/softmax/exp_log_softmax_001.yaml) |
| LogSoftmax 反向 | exp_log_softmax_backward_001 | 25% | 反向梯度：`dy - softmax(x) * sum(dy)` | [yaml](../../raw/experiences/softmax/exp_log_softmax_backward_001.yaml) |
| Softmax 反向（inner） | exp_softmax_backward_001 | 25% | 反向：`p * (dy - dot(p, dy))` | [yaml](../../raw/experiences/softmax/exp_softmax_backward_001.yaml) |
| Softmax 反向（非 inner） | exp_softmax_backward_non_inner_001 | 20% | 非内层维度反向，二维 tile 处理 | [yaml](../../raw/experiences/softmax/exp_softmax_backward_non_inner_001.yaml) |

## 关键技术

### Online Softmax 算法（单趟）

核心思想：维护运行最大值 `m_i` 和运行指数和 `l_i`，一次遍历完成 max+sum：

```
m_new = max(m_old, x_tile_max)
l_new = l_old * exp(m_old - m_new) + sum(exp(x_tile - m_new))
```

最终输出：`softmax(x)_i = exp(x_i - m_final) / l_final`

```python
@triton.jit
def softmax_online_kernel(x_ptr, y_ptr, n_rows, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * n_cols

    # 在线算法：单趟计算 max 和 sum
    m = tl.full([1], float('-inf'), dtype=tl.float32)
    l = tl.zeros([1], dtype=tl.float32)
    for col in range(0, n_cols, BLOCK_N):
        cols = col + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new

    # 第二趟：写出归一化结果
    for col in range(0, n_cols, BLOCK_N):
        cols = col + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        y = tl.exp(x - m) / l
        tl.store(y_ptr + row * n_cols + cols, y.to(tl.float16), mask=mask)
```

### 数值稳定化

fp16/bf16 下 `exp(x)` 在 x > 65504 时溢出。**必须**在 exp 前减去行内最大值：

```python
x_stable = x - tl.max(x, axis=0)  # 防止 exp 溢出
p = tl.exp(x_stable)
p = p / tl.sum(p)
```

参考：[exp_stability_001](../../raw/experiences/general_stability/exp_stability_001.yaml)

### UB 容量限制（N 大于 192KB 时分块）

910b2/910b3 UB 约 192KB。fp16 时单行最大元素数：`192*1024/2 = 98304`。  
超出时使用分块两趟算法（exp_softmax_non_inner_001 / exp_log_softmax_001 的策略）。

## 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| fp16 exp 溢出 | x 不减 max 直接 exp，输出 Inf | 先减行内最大值（数值稳定化） |
| outer dim 效率低 | 非 inner dim 归约需要 gather-like 访问 | 转置后处理内层，或使用 non_inner 变体 |
| 两趟访存带宽瓶颈 | 独立读两次 HBM | 使用 Online Softmax 合并为单趟 |
| Attention 中单独 Softmax | 多余 HBM 写回 | 将 Softmax 融入 Attention 内核（Flash Attention） |

## 相关页面

- [ops/attention.md](attention.md) — Flash Attention 内嵌 Online Softmax，是最重要的 Softmax 融合场景
- [ops/reduction.md](reduction.md) — Reduction 原语与 Softmax 共享归约逻辑
- [ops/loss.md](loss.md) — CrossEntropy/LogSoftmax 常与 Loss 计算融合
- [techniques/online_algorithms.md](../techniques/online_algorithms.md) — Online Softmax 算法详解
- [techniques/numerical_stability.md](../techniques/numerical_stability.md) — 数值稳定性指南
