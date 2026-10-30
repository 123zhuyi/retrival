# moe_init_routing_v2 - 迁移分类

| 知识点 | CANN | Triton | 分类 | 可信度 |
|---|---|---|---|---|
| sort(expertIdx) | moe_v2_sort_*.h | `tl.sort` / scatter | transferable | inferred |
| srcToDstWithCapacity | moe_v2_src_to_dst_with_capacity.h | scatter + capacity 裁剪 | transferable | inferred |
| gather expandedX | moe_v2_gather_out.h | `tl.load` | transferable | inferred |
| expertTokenOut | moe_v2_expert_token_out.h | 计数输出 | transferable | inferred |
| drop_pad_mode | def.cpp L57 | mask/截断 | transferable | inferred |
| expert_capacity/expert_num | def.cpp L55-56 | constexpr 参数 | transferable | inferred |
| 5 tiling key 分派 | kernel.cpp | `if constexpr` | rewrite | inferred |
| core id 切分 | tiling.cpp | `tl.program_id` + `grid` | rewrite | inferred |
| MIX_AIV_1_0 | kernel.cpp | 纯 vector kernel | transferable | inferred |
| ascent310p 支持 | def.cpp L100-101 | 架构约束 | needs verification | inferred |
| 多架构配置差异 | def.cpp config 段 | 编译期选择 | needs verification | inferred |