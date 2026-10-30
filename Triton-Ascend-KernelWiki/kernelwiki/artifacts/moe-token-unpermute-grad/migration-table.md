# moe_token_unpermute_grad - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN 来源 | Triton 处理方式 | 分类 | 可信度 |
|---|---|---|---|---|
| scatter permutedTokensGrad[sortedIndices] | README L38-44 | `tl.store(grad + sorted_idx*stride, val)` | transferable | inferred |
| probsGrad topK 求和 | README L34 | `tl.sum(grad * gathered, axis=topK)` | transferable | inferred |
| 有 probs 加权 scatter | README L38 | `grad * probs` 后 scatter | transferable | inferred |
| 无 probs 纯 scatter | README L44 | `tl.store(grad + sorted_idx, unpermuted_grad)` | transferable | inferred |
| AIV-only | kernel L20-47 | 单 program family | transferable | inferred |
| 4 tiling key 分支 | kernel L30-46 | `if constexpr` | rewrite | inferred |
| 2 模板 ProbNone/ProbNotNone | kernel L31/39 | 独立 kernel / `if constexpr` | rewrite | inferred |
| 950 转调 V2Grad | README L121-129 | 框架层路由 | CANN-specific | source-reported |