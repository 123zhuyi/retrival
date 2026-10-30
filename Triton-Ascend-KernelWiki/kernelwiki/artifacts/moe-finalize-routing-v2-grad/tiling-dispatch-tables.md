# MoeFinalizeRoutingV2Grad - Tiling / 核切分 / 模板 dispatch (S4 + S5)

## A. 算子语义

`MoeFinalizeRoutingV2` 的反向（README.md）。R=batch*sequence, H=hidden, K=topk。

3 场景（按 scales/bias 是否存在）：

1. **无 scales**：`gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i/K][j]`（scatter）
2. **有 scales 无 bias**：
   - `gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i/K][j] * scales[i]`
   - `gradScalesOut[i] = sum(expandedX[expandedRowIdx[i]][j] * gradY[i/K][j])`
3. **有 scales 有 bias**：
   - 同 2
   - `gradScalesOut[i] = sum((expandedX[expandedRowIdx[i]][j] + bias[expertIdx[i]][j]) * gradY[i/K][j])`

## B. Tiling 参数

TilingData 字段（`tiling.h`）：核数、H/K 维、循环控制等。H 轴切分策略（CutH/NotCutH）。

## C. 模板 dispatch（kernel L20-58）

`KERNEL_TYPE_MIX_AIV_1_0`（AIV-only）。4 tiling key：
- `WITHOUT_SCALE_NOT_CUT_H` (10001) - 无 scale，H 不切分
- `WITHOUT_SCALE_CUT_H` (10002) - 无 scale，H 切分
- `WITH_SCALE_NOT_CUT_H` (20001) - 有 scale，H 不切分
- `WITH_SCALE_CUT_H` (20002) - 有 scale，H 切分

4 模板类：`MoeFinalizeRoutingV2Grad{With,Without}Scale{Cut,NotCut}H`

## D. 架构

- ascend910b/910_93（membase, opFile=moe_finalize_routing_v2_grad）
- ascend950（regbase, opFile=moe_finalize_routing_v2_grad_apt）

## E. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 |
|---|---|
| AIV-only | 纯 scatter + reduce |
| scatter gradExpandedX | `tl.atomic_add(gradExpandedX + idx * stride, gradY[i/K]*scale)` |
| gradScales reduce | `tl.sum(expandedX[idx] * gradY[i/K])` |
| H 切分 | BLOCK_H constexpr |
| bias 融合 | `+ tl.load(bias + expertIdx)` |