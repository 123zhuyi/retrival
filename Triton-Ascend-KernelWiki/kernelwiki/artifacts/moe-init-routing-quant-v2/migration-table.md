# moe_init_routing_quant_v2 - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN 来源 | Triton 处理方式 | 分类 | 可信度 |
|---|---|---|---|---|
| keyValueSort | README L27-31 | `tl.sort` / 并行排序 | transferable | inferred |
| expandedRowIdx 反向映射 | README L33-37 | scatter store | transferable | inferred |
| Histogram + Cumsum（dropless） | README L39-43 | `tl.histogram` + prefix sum | transferable | inferred |
| Histogram（drop） | README L45-49 | `tl.histogram` | transferable | inferred |
| 静态 quant round(x·scale+offset) | README L52-56 | `tl.cast(round(...), int8/int4)` | transferable | inferred |
| 动态 quant row_max(abs)/127 | README L58-67 | `tl.max(abs(x)) / 127` + `round(x/dyn)` | transferable | inferred |
| gather expandedX | README L79-83 | `tl.load(quantResult + sorted_idx*stride)` | transferable | inferred |
| drop 容量截断/补零 | README L20 | `if idx < expertCapacity` + zero pad | transferable | inferred |
| AIV-only (MIX_AIV_1_0) | kernel L43 | 单 program family | transferable | inferred |
| INT8/INT4 输出 | def.cpp L49 | dtype constexpr | transferable | inferred |
| tiling key 3 档分支 | kernel L59-146 | `if constexpr`/launcher | rewrite | inferred |
| drop/dropless + 静态/动态 quant 组合 | tiling key 编码 | 模式 constexpr | rewrite | inferred |
| arch32 vs arch35 | def.cpp L84-91 | launcher 选 kernel | rewrite | inferred |
| arch35 regbase/simt | arch35/*.h | 不迁移（CANN 机制） | CANN-specific | inferred |
| fullload 快速路径 | kernel L59-76 | 融合单 kernel | rewrite | inferred |