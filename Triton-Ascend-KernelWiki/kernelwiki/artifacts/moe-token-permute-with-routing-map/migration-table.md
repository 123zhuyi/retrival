# moe_token_permute_with_routing_map - CANN -> Triton-Ascend 迁移分类

四分类（directly transferable / rewrite to Triton-Ascend / CANN-specific do-not-transfer / needs later verification）。MoE Token Permute With Routing Map 是 routingMap 驱动的 token 置换算子。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| maskedSelect（routingMap.T -> sortedIndicesFirst） | README L28；`masked_select_v3.h` | `tl.load(tokens + mask_nonzero_indices)` | transferable | inferred | yes（maskedSelect 精确语义） |
| argSort（sortedIndicesFirst -> sortedIndicesOut） | README L32；`moe_sort_*.h` | `tl.argsort` 或自定义排序 | transferable | inferred | yes（多核归并排序映射） |
| index_select（tokens.index_select(0, sortedIndicesOut)） | README L66；`moe_index_copy.h` | `tl.load(tokens + sorted_indices)` | transferable | inferred | no |
| permuteTokens[sortedIndicesOut[i]] = tokens[i//topK] | README L44 | `tl.store(permuted + sorted_indices, tokens[i//topK])` | transferable | inferred | no |
| probs maskedSelect + index_select | README L48/88；`moe_permute_prob.h` | 同 tokens 路径 | transferable | inferred | no |
| dropAndPad=false（topK 模式） | README L21-49 | `if constexpr` 分支 | rewrite | inferred | no |
| dropAndPad=true（capacity 模式） | README L51-89 | `if constexpr` 分支 | rewrite | inferred | no |
| 单核/多核排序分发（TilingKey 1-8） | kernel.cpp L99-154 | `tl.program_id` 多核 + 单核 fallback | rewrite | inferred | no |
| 3510 填充模式（TilingKey 9/19） | kernel.cpp | arch35 专属分支 | rewrite | inferred | no |
| arch35 SIMD/SIMT 二维 gather | `gather_v2_simd/simt_two_dim.h` | `tl.load` 间接索引 | rewrite | inferred | yes（SIMD/SIMT 精度） |
| 归并排序（moe_mrgsort） | `moe_mrgsort.h` | Triton 不直接映射，用 `tl.argsort` | rewrite | inferred | yes（归并算法等价性） |
| 核间切分 | `tiling.cpp` | `tl.program_id` + grid | rewrite | inferred | no |
| MTE DataCopy 搬运 | kernel 模板 | Triton load/store | CANN-specific | inferred | no |
| 多核归并同步（SyncAll） | kernel.cpp L45/82 | 不迁移；Triton kernel launch barrier | CANN-specific | inferred | no |
| tiling 参数数值 | `tiling.cpp` | constexpr 值 source-reported | needs verification | inferred | yes |
| UB 字节预算 | `tiling.h` | 不声称；需硬件实测 | needs verification | inferred | yes |
| tokens_num/expert_num < 16777215 约束 | README L164 | `tl.constexpr` 约束 | rewrite | inferred | no |
| routingMap 每行 1/true 个数固定且 < 512 | README L165 | 约束 | rewrite | inferred | no |