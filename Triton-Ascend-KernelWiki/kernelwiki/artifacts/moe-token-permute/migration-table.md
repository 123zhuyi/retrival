# moe_token_permute - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN 来源 | Triton 处理方式 | 分类 | 可信度 |
|---|---|---|---|---|
| argSort(indices) | README L18-22 | `tl.sort` / 并行排序 | transferable | inferred |
| argSort(sortedIndicesFirst) | README L24-26 | `tl.sort` | transferable | inferred |
| index_copy permuteTokens[idx]=tokens[i//topK] | README L28-30 | `tl.store` scatter copy | transferable | inferred |
| paddedMode 简化 | README L34-40 | `if constexpr` 分支 | transferable | inferred |
| AIV-only | kernel L47-49 | 单 program family | transferable | inferred |
| 8 tiling key 分支 | kernel L65-81 | `if constexpr` | rewrite | inferred |
| OneCore/MultiCore | tiling.h | `tl.program_id` + grid | rewrite | inferred |
| SpiltD（D 维切分） | moe_index_copy_spilt_d.h | BLOCK_D constexpr | rewrite | inferred |
| 950 转调 V2 | README L105-112 | 框架层路由 | CANN-specific | source-reported |