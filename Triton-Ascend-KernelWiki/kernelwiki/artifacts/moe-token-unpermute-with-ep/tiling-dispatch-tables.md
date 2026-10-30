# MoeTokenUnpermuteWithEp - Tiling / 核切分 / 模板 dispatch (S4 + S5)

## A. 算子语义

根据 sortedIndices 的下标位置，从 permutedTokens 取数据（与 probs 相乘），合并累加（README.md）：

1. `sortedIndices = sortedIndices[range[0] <= i < range[1]]` - range 过滤
2. `permutedTokens = permutedTokens.indexSelect(0, sortedIndices)` - gather
3. 若 probs 非 None：`permutedTokens_k = permutedTokens_k * probs_{i,j}` - 乘 probs
4. `out_i = Σ_{k=i*numTopk}^{(i+1)*numTopk-1} permutedTokens_k` - 跨 topK reduce sum

## B. 输入输出（def.cpp）

- **输入**：permuted_tokens（BF16/FP16/FLOAT）、sorted_indices（INT32）、probs（可选）
- **输出**：unpermuted_tokens（BF16/FP16/FLOAT）
- **属性**：num_topk(1)、range([0,0])、padded_mode(false)、restore_shape（仅 nullptr）

## C. 架构

ascend910b/910_93 + kirinX90/kirin9030（不支持 ascend950）

## D. 模板 dispatch（kernel L39-49）

4 tiling key：
- `0`：无 probs
- `1`：float probs
- `2`：half probs
- `3`：bf16 probs（排除 arch 3003/3113）

1 模板类 `KernelMoeTokenUnpermuteWithEp`，通过模板参数 `haveProbs` + probs dtype 区分。

## E. 约束（README.md）

- topK_num <= 512
- 不支持 paddedMode=True
- rangeOptional 为空时回退到 `MoeTokenUnpermute`

## F. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 |
|---|---|
| AIV-only | 纯 gather + reduce |
| indexSelect | `tl.load(permutedTokens + idx * stride)` |
| 乘 probs | `* probs[i,j]` |
| sum | `tl.sum` 跨 topK 维 |
| range 过滤 | mask |