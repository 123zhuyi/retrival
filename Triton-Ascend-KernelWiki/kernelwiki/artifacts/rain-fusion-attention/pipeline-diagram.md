# rain_fusion_attention - 流水编排分析

## 1. 核心计算

selectIdx 驱动的块级稀疏 FlashAttention（推理）：

```
for each Q block (blockShapeX):
  load Q_block [BLOCK_M, D]
  load selectIdx[block][headNum] -> selected KV blocks
  m_i = -inf, l_i = 0, acc = 0
  for each selected KV block:
    load K_block [BLOCK_N, D]
    s = Q_block · K_block^T × scale  [BLOCK_M, BLOCK_N]
    s = mask(s, atten_mask)          # 可选
    # online softmax + PV rescale
    m_ij = max(s)
    s = s - m_ij;  p = exp(s)
    l_ij = sum(p)
    alpha = exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha + p · V_block
    m_i = max(m_i, m_ij)
  attention_out = acc / l_i
  softmaxLse = log(l_i) + m_i        # 可选
```

## 2. 流水阶段

| 阶段 | 单元 | 操作 |
|---|---|---|
| 1. Q load | AIC | Q 块搬入 |
| 2. selectIdx | AIC | 读稀疏块索引 |
| 3. K load | AIC | 选中 KV 块搬入 |
| 4. QK | AIC | Q·K^T |
| 5. softmax | AIV | online softmax（FP32/Half） |
| 6. PV | AIC | P·V |
| 7. rescale+写回 | AIV | ÷sum + LSE |

## 3. 阶段 -> Triton 抽象

全部可迁移（这是标准稀疏 FlashAttention 流程，与 SFA 类似）。CATLASS 模板库机制为 CANN 专有。