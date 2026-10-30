# moe_finalize_routing - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN 来源 | Triton 处理方式 | 分类 | 可信度 |
|---|---|---|---|---|
| MoE 合并公式 gather+scale+add | README L21-27 | `tl.load` gather + 乘加累加 | transferable | inferred |
| expandedRowIdx 离散索引 | def.cpp L52-57 | `tl.load(idx)` 间接寻址 | transferable | inferred |
| bias[expertId] 专家偏置 | def.cpp L40-45 | `tl.load(bias + expert*stride)` | transferable | inferred |
| scales 加权 | def.cpp L46-51 | 元素乘 | transferable | inferred |
| x1 + x2Optional 跳跃连接 | def.cpp L28-38 | 元素加 | transferable | inferred |
| FP32/FP16/BF16 dtype | def.cpp L24 | dtype constexpr | transferable | inferred |
| 19 tiling key 分支 | kernel L43-142 | `if constexpr`/launcher | rewrite | inferred |
| BUFFER_NUM=1 / PARALLEL_NUM=2 | common.h L26-27 | 不迁移（CANN 优化） | CANN-specific | inferred |
| 310P BF16 排除 | kernel L54/61 `#if` | 编译期排除 | CANN-specific | inferred |
| Kirin 仅 FP16/FLOAT | def.cpp L97-143 | dtype 分支 | rewrite | inferred |