# moe_token_permute_grad - CANN -> Triton-Ascend 迁移分类

四分类（directly transferable / rewrite to Triton-Ascend / CANN-specific do-not-transfer / needs later verification）。MoE Token Permute Grad 是 token 置换的反向（即 unpermute）。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| inputGrad = permutedOutputGrad.indexSelect(0, sortedIndices) | README L20 | `tl.load(permuted_grad + sorted_indices)` | transferable | inferred | no |
| reshape(-1, numTopk, hiddenSize) | README L24 | `tl.reshape` | transferable | inferred | no |
| sum(dim=1) 沿 numTopk 轴 | README L28 | `tl.sum(x, axis=1)` | transferable | inferred | no |
| unpermute = indexSelect + reshape + sum | kernel.cpp L31-41 | `tl.load` + `tl.sum` | transferable | inferred | no |
| TilingKey 按 dtype（0/2/4） | kernel.cpp L30-42 | `if constexpr` / dtype 模板 | rewrite | inferred | no |
| 950 转调用 MoeInitRoutingV2Grad | README L84-90 | 不迁移（CANN 框架代理） | CANN-specific | inferred | yes（代理正确性） |
| numTopk <= 512 约束 | README L83 | `tl.constexpr` 约束 | rewrite | inferred | no |
| padded_mode 属性 | def.cpp L38 | `tl.constexpr` | rewrite | inferred | no |
| 核间切分 | `tiling.cpp` | `tl.program_id` + grid | rewrite | inferred | no |
| MTE DataCopy 搬运 | kernel 模板 | Triton load/store | CANN-specific | inferred | no |
| tiling 参数数值 | `tiling.cpp` | constexpr 值 source-reported | needs verification | inferred | yes |
| UB 字节预算 | `tiling.h` | 不声称；需硬件实测 | needs verification | inferred | yes |