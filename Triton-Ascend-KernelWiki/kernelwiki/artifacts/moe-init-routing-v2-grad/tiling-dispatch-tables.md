# MoeInitRoutingV2Grad - Tiling / 核切分 / 模板 dispatch (S4 + S5)

## A. 算子语义

`MoeInitRoutingV2` 的反向（README.md L18-27）：

```
gradX_i = Σ_{t=0}^{topK} gradExpandedX[expandedRowIdx[i*topK + t]]
```

按 expandedRowIdx 反向 gather gradExpandedX，跨 topK 维做 reduce sum。

## B. Tiling 参数

TilingData 字段（`tiling.h` L40-49）：
- `coreNum/n/e/c/cols/k/activeNum` - 基本参数
- `MoeV2GradComputeParams`：`needCoreNum/perCoreElements/lastCoreElements/elementCopyLoops/elementPerCopyCols/binaryAddBufferNum/tmpBufferNum/exponentOfBinary/copyBufferSize/tokensFormer/perCoreTokensLoop/...`（核循环与二路归并）

arch35 三套 TilingData：
- `MoeInitRoutingV2GradRegbaseSplitHTilingData`（200xxx）：H 轴切分，`hBlockFactor/hUbFactor/kUbFactor`
- `MoeInitRoutingV2GradRegbaseFullLoadTilingData`（300xxx）：全加载，`nBlockFactor/nUbFactor/hUbFactor`
- `MoeInitRoutingV2GradRegbaseTilingData`（400xxx）：通用，`nBlockFactor/nUbFactor/kUbFactor/hUbFactor`

## C. 模板 dispatch（kernel L47-83）

9 个 tiling key，`KERNEL_TYPE_AIV_ONLY`（g_coreType==AIC return）：
- **3 模式**：Dropless（1000/1001/1002）、Activate（1010/1011/1012）、PositionPad0（1100/1101/1102）
- **3 dtype**：FLOAT/FLOAT16/BF16

3 个模板类：`MoeInitRoutingV2GradDroplessCompute` / `MoeInitRoutingV2GradActivateCompute` / `MoeInitRoutingV2GradPositionPad0Compute`

## D. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 |
|---|---|
| AIV-only | 纯 gather + reduce |
| expandedRowIdx gather | `tl.load(gradExpandedX + expandedRowIdx[i*topK+t] * stride)` |
| 跨 topK 累加 | 循环 `+=` 或 `tl.sum` |
| 核间切分 | grid 1D 映射 |
| H 轴切分（arch35 SplitH） | BLOCK_H constexpr |