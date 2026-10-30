# moe_token_permute_with_ep - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN 来源 | Triton 处理方式 | 分类 | 可信度 |
|---|---|---|---|---|
| argSort(indices) | README L22-24 | `tl.sort` / 并行排序 | transferable | inferred |
| argSort(sortedIndicesFirst) | README L26-28 | `tl.sort` | transferable | inferred |
| range 范围切片 | README L30-34 | `if range[0]<=idx<range[1]` | transferable | inferred |
| index_copy tokens | README L33-34 | `tl.store` scatter copy | transferable | inferred |
| index_copy probs | README L37-38 | `tl.store` scatter copy（双输出） | transferable | inferred |
| paddedMode 简化 | README L42-48 | `if constexpr` 分支 | transferable | inferred |
| AIV-only | kernel L51-53 | 单 program family | transferable | inferred |
| 8 tiling key 分支 | kernel L69-85 | `if constexpr` | rewrite | inferred |
| OneCore/MultiCore | tiling.h | `tl.program_id` + grid | rewrite | inferred |
| SpiltD | moe_index_copy_spilt_d.h | BLOCK_D constexpr | rewrite | inferred |
| range 空回退 token_permute | README L133 | 框架层路由 | CANN-specific | source-reported |