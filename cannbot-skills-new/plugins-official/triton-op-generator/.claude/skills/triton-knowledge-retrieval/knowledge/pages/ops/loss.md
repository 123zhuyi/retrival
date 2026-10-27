---
title: Loss 函数算子优化
category: ops
tags: [loss, cross-entropy, celoss, entropy, label-smoothing, backward-fusion]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# Loss 函数算子优化

## 概述

Loss 函数将模型输出（logits）与标签（labels）对比，计算标量损失值及梯度。常见 Loss 包括：
- **CrossEntropy Loss**：`CE = -log(softmax(x)[label])`，语言模型最常用
- **Label-Smoothed CE**：`CE_smooth = (1-ε)·CE + ε/V·H(uniform)`
- **GRPO Loss**：强化学习策略梯度，含 LogSumExp 归约
- **Entropy Loss**：`H = -Σ p·log(p)`，用于 KL 散度计算

核心优化机会：
1. **前向-反向融合**：CE Loss 的前向和反向 grad 可在同一 kernel 内计算，消除激活重存
2. **Online Softmax + Loss 融合**：GEMM 输出不写 HBM，直接在线计算 softmax + loss
3. **数值稳定化**：log-sum-exp trick 防止 exp 溢出

## 检测标准

满足以下条件归类为 Loss：
- PyTorch 原语：`F.cross_entropy`、`F.nll_loss`、`nn.CrossEntropyLoss`
- 输出为标量（或 per-sample 向量）损失值
- 通常含对 vocab_size 或 class 维度的归约

## 适用条件

| 条件 | 说明 |
|------|------|
| 数据类型 | fp16、bf16（logits）；int32/int64（labels） |
| vocab_size | ≥256 时分块归约；>4096 时建议 online 算法 |
| batch_size | ≥1，通常 n_tokens = batch × seq_len |

## 经验路由

| 场景 | 经验ID | 提升% | 关键技术 | 经验文件 |
|------|--------|-------|----------|----------|
| CrossEntropy 前向（标准） | exp_crossentropy_001 | 35% | Online Softmax，2趟降为1趟；前向反向融合 | [yaml](../../raw/experiences/loss/exp_crossentropy_001.yaml) |
| CrossEntropy 反向 | exp_crossentropy_backward_001 | 30% | 反向梯度 `p - one_hot`，与前向融合 | [yaml](../../raw/experiences/loss/exp_crossentropy_backward_001.yaml) |
| CE Loss（celoss 基础版） | exp_celoss_001 | 30% | log-sum-exp 数值稳定，mask padding | [yaml](../../raw/experiences/loss/exp_celoss_001.yaml) |
| CE Loss 反向 | exp_celoss_bwd_001 | 25% | CE 反向 grad 计算 | [yaml](../../raw/experiences/loss/exp_celoss_bwd_001.yaml) |
| Label Smoothing CE | exp_celoss_smooth_001 | 25% | (1-ε)·CE + ε/V·uniform，单核完成 | [yaml](../../raw/experiences/loss/exp_celoss_smooth_001.yaml) |
| Label Smoothing CE 反向 | exp_celoss_smooth_bwd_001 | 25% | smooth CE 反向 | [yaml](../../raw/experiences/loss/exp_celoss_smooth_bwd_001.yaml) |
| Entropy 反向（通用） | exp_entropy_backward_001 | 25% | 分布熵的梯度 `-log(p) - 1` | [yaml](../../raw/experiences/loss/exp_entropy_backward_001.yaml) |
| Entropy 反向（权重梯度） | exp_entropy_backward_d_weight_001 | 25% | 按 weight 求熵梯度 | [yaml](../../raw/experiences/loss/exp_entropy_backward_d_weight_001.yaml) |
| Entropy 反向（融合版） | exp_entropy_backward_fused_001 | 35% | 前向反向同核，消除激活重存 | [yaml](../../raw/experiences/loss/exp_entropy_backward_fused_001.yaml) |
| Entropy 反向（NPU 专用） | exp_entropy_backward_npu_001 | 35% | Ascend 910b2/b3 路径优化 | [yaml](../../raw/experiences/loss/exp_entropy_backward_npu_001.yaml) |
| Entropy epilogue | exp_entropy_epilogue_001 | 20% | Loss 后处理（归一化/缩放） | [yaml](../../raw/experiences/loss/exp_entropy_epilogue_001.yaml) |
| GEMM + Entropy 融合 | exp_fused_entropy_001 | 45% | hidden@weight.T + softmax + entropy 三合一，消除 logits HBM | [yaml](../../raw/experiences/loss/exp_fused_entropy_001.yaml) |

## 关键技术

### 前向-反向融合（CrossEntropy）

CE 反向梯度 `dL/dx_i = p_i - y_i`（`p_i` = softmax 输出，`y_i` = one-hot）。  
可在一次 kernel 中同时计算 CE Loss 值和梯度，写入 `X_ptr`（OVERWRITE_GRAD_LOGITS 模式）：

```python
@triton.jit
def cross_entropy_fwd_bwd_kernel(X_ptr, Y_ptr, label_ptr, loss_ptr,
                                  n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    label = tl.load(label_ptr + row)
    x_row = X_ptr + row * n_cols

    # Online Softmax：单趟求 max+sum
    m, l = tl.full([1], float('-inf'), tl.float32), tl.zeros([1], tl.float32)
    for col in range(0, n_cols, BLOCK_N):
        cols = col + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new

    # 计算 Loss = -x[label] + log(l) + m
    loss = -tl.load(x_row + label).to(tl.float32) + tl.log(l) + m
    tl.store(loss_ptr + row, loss)

    # 写反向梯度：p - one_hot
    for col in range(0, n_cols, BLOCK_N):
        cols = col + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        p = tl.exp(x - m) / l
        dy = p - (cols == label).to(tl.float32)
        tl.store(Y_ptr + row * n_cols + cols, dy.to(tl.float16), mask=mask)
```

### GEMM + Entropy 完全融合

当 Loss 前置一个大 GEMM（`hidden @ weight.T`，输出 logits 大小为 `n_tokens × vocab_size`）时：  
不将 logits 写到 HBM，在 GEMM tile 循环内直接维护 online softmax 状态，最终计算 loss。  
消除中间 `n_tokens × vocab_size` 矩阵的 HBM 读写，节省大量带宽。

参考：[exp_fused_entropy_001](../../raw/experiences/loss/exp_fused_entropy_001.yaml)

### log-sum-exp 数值稳定

```python
# 标准（可能溢出）
loss = -x[label] + tl.log(tl.sum(tl.exp(x)))

# 稳定版
m = tl.max(x)
loss = -x[label] + m + tl.log(tl.sum(tl.exp(x - m)))
```

## 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| exp 溢出 | fp16 下 logits 未减 max 直接 exp | log-sum-exp trick（减 max 后 exp） |
| 前后向分离 | CE 前向保存 softmax 激活，反向重读 | 前向反向融合，消除激活存储 |
| vocab_size 超 UB | 单次无法加载全行 | 分块两趟（accumulate max, then sum） |
| label padding | padding token 的 loss 错误纳入均值 | 用 mask 过滤 ignore_index 位置 |

## 相关页面

- [ops/softmax.md](softmax.md) — Softmax 是 CE Loss 的核心子算子
- [ops/reduction.md](reduction.md) — 归约原语（LogSumExp 等）
- [techniques/online_algorithms.md](../techniques/online_algorithms.md) — Online Softmax 算法
- [techniques/operator_fusion.md](../techniques/operator_fusion.md) — GEMM+Loss 融合设计
