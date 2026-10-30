# moe_token_permute_grad - 流水编排分析

来源：`op_kernel/moe_token_permute_grad.cpp`（43 行）+ `moe_token_permute_grad.h`（377 行）。MoE Token Permute Grad 是 token 置换的反向（unpermute），纯 Vector 计算。

## 1. 核心计算公式

```
inputGrad = permutedOutputGrad.indexSelect(0, sortedIndices)   # 按 sortedIndices gather
inputGrad = inputGrad.reshape(-1, numTopk, hiddenSize)         # reshape 为 [row, K, H]
inputGrad = inputGrad.sum(dim=1)                                # 沿 K 轴 reduce
```

即：对每个原始 token，将其被置换到 numTopk 个专家位置的梯度求和回原始位置。

## 2. 单核流程

```
for each token row in output:
    for k in 0..numTopk:
        idx = sortedIndices[row * numTopk + k]      # 置换后的位置
        acc += permutedOutputGrad[idx]              # gather + 累加
    inputGrad[row] = acc
```

## 3. 950 代理关系

950 架构不直接实现，转调用 `aclnnMoeInitRoutingV2Grad`：
- 语义等价：unpermute（indexSelect + sum）≈ MoeInitRoutingV2Grad 的 scatter-add 反向
- 参数映射见 tiling_dispatch_tables.md §A

## 4. 阶段 -> Triton 抽象

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移 |
|---|---|---|---|
| indexSelect | sortedIndices + DataCopy | `tl.load(permuted_grad + sorted_indices)` | 是（语义） |
| reshape | view 变换 | `tl.reshape` | 是 |
| sum(dim=1) | ReduceSum | `tl.sum(x, axis=1)` | 是（数学） |
| MTE 搬运 | DataCopyPad | `tl.load`/`tl.store` | 否（机制） |
| 950 代理 | 框架转调用 | 不迁移 | 否（机制） |