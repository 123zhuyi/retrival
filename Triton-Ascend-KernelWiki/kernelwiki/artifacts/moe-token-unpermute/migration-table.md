# moe_token_unpermute - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN 来源 | Triton 处理方式 | 分类 | 可信度 |
|---|---|---|---|---|
| gather permutedTokens[sortedIndices] | README L23-25 | `tl.load(permuted + sorted_idx*stride)` | transferable | inferred |
| probs 乘 | README L27-29 | `tl.load(probs) * gathered` | transferable | inferred |
| topK 累加求和还原 | README L31-33 | `tl.sum(gathered.reshape(topK), axis=0)` | transferable | inferred |
| 无 probs topK=1 简化 | README L37-47 | `if constexpr` 分支 | transferable | inferred |
| AIV-only | kernel L27-88 | 单 program family | transferable | inferred |
| ~16 tiling key 分支 | kernel L33-87 | `if constexpr` | rewrite | inferred |
| mix mode（probs dtype ≠ token dtype） | tiling key 11/13/17/21/25/27 | dtype constexpr 分支 | rewrite | inferred |
| 310P/Kirin 仅 FP16/FP32 | def.cpp L78-118 | dtype 分支 | rewrite | inferred |
| 950 转调 finalize_routing_v2 | README L109-115 | 框架层路由 | CANN-specific | source-reported |