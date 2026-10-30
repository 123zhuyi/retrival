# moe_finalize_routing_v2_grad - 流水编排分析

来源：`op_kernel/moe_finalize_routing_v2_grad.cpp` + 4 个模板文件。MoE Finalize Routing V2 Grad 是路由最终化的反向传播，核心是 scatter-add + 可选 scale 梯度。

## 1. 核心计算公式

```
场景 1（无 scale）：
  gradExpandedXOut[expandedRowIdx[i]][j] += gradY[i/K][j]

场景 2（有 scale，无 bias）：
  gradExpandedXOut[expandedRowIdx[i]][j] += gradY[i/K][j] * scales[i]
  gradScalesOut[i] = sum(expandedX[expandedRowIdx[i]][j] * gradY[i/K][j])

场景 3（有 scale + bias）：
  gradExpandedXOut[expandedRowIdx[i]][j] += gradY[i/K][j] * scales[i]
  gradScalesOut[i] = sum((expandedX[expandedRowIdx[i]][j] + bias[expertIdx[i]][j]) * gradY[i/K][j])
```

## 2. 单核流程

```
for each (i, j) in (R*K) × H:
    idx = expandedRowIdx[i]
    gradExpandedX[idx][j] += gradY[i/K][j] * scales[i]    # atomic_add
    if with_scale:
        gradScales[i] += expandedX[idx][j] * gradY[i/K][j]  # 累加
```

## 3. 阶段 → Triton 抽象

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移 |
|---|---|---|---|
| scatter-add | atomic_add + DataCopyPad | `tl.atomic_add` | 是（语义） |
| gradScales 累加 | Mul + ReduceSum | `tl.sum(expandedX[idx] * gradY)` | 是（数学） |
| 3 场景分支 | if constexpr + TilingKey | `if constexpr` | 是 |
| MTE 搬运 | DataCopyPad | `tl.load`/`tl.store` | 否（机制） |