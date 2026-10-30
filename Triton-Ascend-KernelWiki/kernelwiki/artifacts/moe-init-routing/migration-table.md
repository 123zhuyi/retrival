# moe_init_routing - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN | Triton | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| 公式：sort(expertIdx, rowIdx) | moe_sort_*.h | `tl.sort` 或 scatter | directly transferable | inferred | no |
| 公式：expandedRowIdx[sortedRowIdx[i]]=i | moe_src_to_dst_op.h | scatter 逆映射 | directly transferable | inferred | no |
| 公式：expandedX[i]=x[sortedRowIdx[i]%numRows] | moe_gather_out.h | `tl.load(x + idx*cols)` | directly transferable | inferred | no |
| activeNum 控制有效行数 | def.cpp L56 | mask 或 slice | directly transferable | inferred | no |
| 5 tiling key 分派（0/1/2/3/4） | kernel.cpp L48-99 | `if constexpr` / 独立 kernel | rewrite | inferred | no |
| 单核/多核排序策略 | moe_sort_*.h | Triton 单 kernel 内并行 | rewrite | inferred | no |
| 小活跃行 gather 优化 | moe_gather_out_small_activate_row.h | 运行时分支 | rewrite | inferred | no |
| core id（N×K 切分） | tiling.cpp | `tl.program_id` + `grid` | rewrite | inferred | no |
| MIX_AIV_1_0（纯 Vector） | kernel.cpp L31 | 纯 vector kernel | directly transferable | inferred | no |
| TQue / MTE / 手动同步 | - | 不迁移 | CANN-specific | inferred | no |