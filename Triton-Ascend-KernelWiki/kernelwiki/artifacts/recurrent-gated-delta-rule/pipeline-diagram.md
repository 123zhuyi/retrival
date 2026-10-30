# recurrent_gated_delta_rule - 流水编排分析

## 1. 核心公式

```
S_t = α·Diag(αk)·S_{t-1} + β·(v_t - α·Diag(αk)·S_{t-1}·k_t)·k^T_t
o_t = S_t·q_t / √d
```
（README L24-31）。其中 `α = e^g`，`αk = e^gk`（README L89-100）。

## 2. 循环状态更新流程

```
for each time step t:
  # 门控
  α = exp(g), αk = exp(gk)
  S_prev = S[t-1]  # [dv, dk]
  k = k[t], v = v[t], q = q[t]  # 当前步输入

  # 衰减记忆
  decayed = α · Diag(αk) · S_prev        # [dv, dk]
  # 误差项
  error = v - decayed · k                # [dv] (S_prev·k 是 [dv] 向量)
  # 更新
  S_t = decayed + β · error · k^T        # [dv, dk]
  # 输出
  o_t = S_t · q / √d                      # [dv]
```

## 3. 流水阶段

| 阶段 | 操作 | 说明 |
|---|---|---|
| 1. Init | 读 tiling data，置全局 tensor 指针 | kernel.h L72 |
| 2. Process 循环 | 遍历 batch×token，更新 S_t 并输出 o_t | kernel.h Process |
| 2a. 门控计算 | exp(g)/exp(gk)，衰减 S_prev | 元素级 |
| 2b. 误差计算 | v - decayed·k | 向量点积 |
| 2c. 状态更新 | decayed + β·error·k^T | 外积 |
| 2d. 输出 | S_t·q / √d | 矩阵-向量乘 |

## 4. 阶段 -> Triton 抽象

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移 |
|---|---|---|---|
| 门控 exp(g) | `MathExp` | `tl.exp` | 是 |
| 记忆衰减 α·Diag(αk)·S_prev | 元素乘 + 矩阵乘 | `tl.dot` | 是 |
| 误差 (v - decayed·k) | 向量点积 + 减 | `tl.dot` + `tl.where` | 是 |
| 状态更新 β·error·k^T | 外积 | `tl.dot(error, k)` | 是 |
| 输出 S_t·q·scale | 矩阵-向量乘 | `tl.dot(S, q) * scale` | 是 |
| AIV-only 无 Cube | 纯 Vector | 单 program family | 是 |
| BUFFER_NUM=1 | 单 buffer | 不迁移（编译器调度） | 否 |
| 变长 actual_seq_lengths | cuSeqlens 前缀和 | 逐 token 步进 | 是 |