# moe_token_permute_grad - tiling 与 dispatch 分析

来源：`moe/moe_token_permute_grad/`（commit `d28d8f9…`）。MoE Token Permute Grad 是 token 置换的反向（unpermute）。

## A. 架构分发

`def.cpp` L46-47 注册 AICore config：

| 架构 | SoC | config | kernel 入口 | 说明 |
|---|---|---|---|---|
| Atlas A2 训练/推理 | `ascend910b` L46 | `moe_token_permute_grad` | `moe_token_permute_grad.cpp` | 直接实现 |
| Atlas A3 训练/推理 | `ascend910_93` L47 | 同上 | 同上 | 直接实现 |
| Ascend 950PR/950DT | - | - | - | **转调用 `aclnnMoeInitRoutingV2Grad`**（框架代理） |

> 950 不直接实现，框架内部转调用 MoeInitRoutingV2Grad，参数映射：
> - permutedOutputGrad -> gradExpandedX
> - sortedIndices -> expandedRowIdx
> - numTopk -> topK
> - paddedMode -> dropPadMode

## B. TilingKey 与 dispatch（kernel.cpp L30-42）

| TilingKey | dtype | 模板 |
|---|---|---|
| 0 | BF16 | `KernelMoeTokenUnpermute<bfloat16_t, int32_t, bfloat16_t, false>` |
| 2 | FP16 | `KernelMoeTokenUnpermute<half, int32_t, half, false>` |
| 4 | FP32 | `KernelMoeTokenUnpermute<float, int32_t, float, false>` |

> probs 模板参数始终 false。

## C. 核间切分

| 项 | 结论 |
|---|---|
| 逻辑轴 | 沿 token 维（permuted 维）切分 |
| 切分方式 | 按 aivNum 沿 row 轴外切 |

## D. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| indexSelect | `tl.load(permuted_grad + sorted_indices)` | 间接 load |
| reshape(-1, K, H) | `tl.reshape` | 静态 shape |
| sum(dim=1) | `tl.sum(x, axis=1)` | 沿 K 轴 reduce |
| TilingKey dtype | `if constexpr` / dtype 模板 | 不迁移 key |
| 950 代理 | 不迁移 | CANN 框架机制 |