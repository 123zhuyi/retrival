# MoeTokenPermuteWithEpGrad - Tiling / 核切分 / 模板 dispatch (S4 + S5)

## A. 算子语义

`MoeTokenPermuteWithEp` 的反向（README.md）：

1. `sortedIndices = sortedIndices[range[0] <= i < range[1]]` - range 过滤
2. `tokenGradOut = permutedTokensOutputGrad.indexSelect(0, sortedIndices)` - gather
3. `tokenGradOut = tokenGradOut.reshape(-1, topK, hiddenSize)` - reshape
4. `tokenGradOut = tokenGradOut.sum(dim=1)` - 跨 topK reduce sum

## B. 输入输出（def.cpp）

- **输入**：permuted_tokens_output_grad（BF16/FP16/FLOAT）、sorted_indices（INT32）、permuted_probs_output_grad（可选）
- **输出**：token_grad、probs_grad（BF16/FP16/FLOAT）
- **属性**：num_topk（必需）、range（[0,0] 默认，ep 切分范围）、padded_mode（false，仅支持 false）

## C. 架构

仅 ascend910b/910_93（不支持 ascend950，README.md L7）

## D. 模板 dispatch（kernel L53-59）

`KERNEL_TYPE_AIV_ONLY`。2 tiling key：
- `0`：无 probs（仅计算 token_grad）
- `1`：有 probs（计算 token_grad + probs_grad）

1 个模板类 `KernelMoeTokenUnpermuteWithEp`，通过 `haveProbs` 参数区分。

## E. 约束（README.md L108-110）

- top_k <= 512
- 不支持 paddedMode=True
- rangeOptional 为空时回退到 `MoeTokenPermuteGrad`

## F. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 |
|---|---|
| AIV-only | 纯 gather + reduce |
| indexSelect | `tl.load(grad + sortedIndices[i] * stride)` |
| reshape | view（无搬运）|
| sum(dim=1) | `tl.sum` 跨 topK 维 |
| range 过滤 | mask |