# moe_compute_expert_tokens - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN 来源 | Triton 处理方式 | 分类 | 可信度 |
|---|---|---|---|---|
| BinarySearch(sortedExperts, numExperts) | README L21-23 | 自定义并行二分查找 / `tl.where` 累加 | transferable | inferred |
| sortedExperts 已排序前提 | README L39 | 前提（来自 init_routing） | transferable | inferred |
| 累积 token 数输出 | README L16 | `out[i] = 累积行数` | transferable | inferred |
| AIV-only (MIX_AIV_1_0) | kernel L24 | 单 program family | transferable | inferred |
| 3 tiling key 分支 | kernel L32-44 | `if constexpr` | rewrite | inferred |
| SS/M/L 模板选择 | tiling.h | 数据量选择策略 | rewrite | inferred |