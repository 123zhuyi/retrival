# moe_init_routing_quant - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN 来源 | Triton 处理方式 | 分类 | 可信度 |
|---|---|---|---|---|
| keyValueSort(expertIdx,rowIdx) | README L19-21 | `tl.sort` / 并行排序 | transferable | inferred |
| expandedRowIdx 反向映射 | README L23-25 | `tl.store(idx*stride, i)` scatter | transferable | inferred |
| gather x[sortedRowIdx] | README L27-29 | `tl.load(x + sorted_idx*stride)` | transferable | inferred |
| quant to INT8 (scale+offset) | def.cpp L57-58 | `tl.cast(round(x*scale+offset), tl.int8)` | transferable | inferred |
| AIV-only | kernel L36-38 | 单 program family | transferable | inferred |
| 4 tiling key 分支 | kernel L53-80 | `if constexpr` | rewrite | inferred |
| OneCore/MultiCore 排序 | tiling.h | `tl.program_id` + grid | rewrite | inferred |
| SmallActiveRow 变体 | kernel L76-80 | 边界处理 | rewrite | inferred |