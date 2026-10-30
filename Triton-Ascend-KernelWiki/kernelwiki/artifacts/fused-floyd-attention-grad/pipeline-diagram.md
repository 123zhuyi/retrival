# fused_floyd_attention_grad - 流水编排分析

来源：`op_kernel/fused_floyd_attention_grad.cpp`（122 行）+ `kernel` 三文件（pre.h/s1s2_bn2gs1s2.h/post.h）。Floyd Attention Grad 是 Floyd Attention 的反向，双流注意力（K1/V1+K2/V2），6 个 matmul 反向。

## 1. 核心计算公式

```
S = softmax(scale * (Q·K1^T + Q·K2^T), mask)    # 前向 P
dS = S * dY                                       # 下游梯度
dV1 = P^T · dY                                    # Value1 梯度
dV2 = P^T · dY                                    # Value2 梯度
dQ = (dS·K1 + dS·K2) / √d                        # Query 梯度
dK1 = (dS^T · Q) / √d                             # Key1 梯度
dK2 = (dS^T · Q) / √d                             # Key2 梯度
```

## 2. 三段式 kernel 结构

```
┌─────────────────────────────────────────────────────────────────┐
│ Pre (dS = softmax * dY)                                        │
│   → 利用前向保存的 softmax_max/sum 恢复 P，再乘以 dY 得 dS    │
│   → SyncALLCores()                                             │
├─────────────────────────────────────────────────────────────────┤
│ S1S2 (6 个 matmul 反向)                                         │
│   → Bmm1:  dV1 = P^T · dY                                      │
│   → Bmm2:  dV2 = P^T · dY                                      │
│   → MM1:   dQ = (dS·K1 + dS·K2) / √d                          │
│   → MM2:   dK1 = (dS^T · Q) / √d                               │
│   → MM3:   dK2 = (dS^T · Q) / √d                               │
│   → SyncALLCores()                                             │
├─────────────────────────────────────────────────────────────────┤
│ Post (梯度搬运 + 格式转换)                                      │
│   → 将 dQ/dK1/dV1/dK2/dV2 从中间格式搬运到输出 GM              │
│   → 可选 ND↔NZ 格式转换                                         │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 双流注意力特色

Floyd Attention 将 seq 作为 batch 轴，将 Q·K^T 分解为 Q·K1^T + Q·K2^T 两部分：
- K1/V1 和 K2/V2 共享 Q
- 梯度计算时 dQ 和 dK1/dK2 各自独立但公式对称
- 每对 K/V 的梯度计算相互独立，可并行

## 4. 阶段 → Triton 抽象

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移 |
|---|---|---|---|
| Pre (dS = S * dY) | Vector elemwise | `dS = S * dY` | 是（数学） |
| dV1/dV2 = P^T · dY | Cube mm | `tl.dot(P.T, dY)` | 是（数学） |
| dQ = (dS·K1 + dS·K2) / √d | Cube mm × 2 + add | `(tl.dot(dS,K1) + tl.dot(dS,K2)) / sqrt(d)` | 是（数学） |
| dK1 = dS^T · Q / √d | Cube mm | `tl.dot(dS.T, Q) / sqrt(d)` | 是（数学） |
| dK2 = dS^T · Q / √d | Cube mm | `tl.dot(dS.T, Q) / sqrt(d)` | 是（数学） |
| 三段式 SyncAll | CrossCoreSync | 不迁移（Triton kernel launch） | 否（机制） |
| ND↔NZ 格式转换 | DataCopyPad | Triton layout（编译器） | 否（机制） |