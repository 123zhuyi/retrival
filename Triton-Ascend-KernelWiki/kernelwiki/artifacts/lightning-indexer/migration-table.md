# lightning_indexer - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN | Triton | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| 公式：Indices = Top-k([1]@[(W@[1])⊙ReLU(Q·K^T)]) | README L20-24 | `tl.dot` + ReLU + weights scale + topK | directly transferable | inferred | no |
| ReLU(Q·K^T) | service_vector.h DoScale | `tl.maximum(s, 0)` | directly transferable | inferred | no |
| weights 缩放 | service_vector.h DoScale | `s *= weight` | directly transferable | inferred | no |
| topK 选择 | service_vector.h ProcessVec | `tl.argsort` | directly transferable | inferred | no |
| sparse_indices 输出 | def.cpp L53-56 | `tl.store(indices_ptr, idx)` | directly transferable | inferred | no |
| sparse_values 输出 | def.cpp L57-60 | `tl.store(values_ptr, val)` | directly transferable | inferred | no |
| sparseMode=3 causal mask | common.h L163 | `tl.where` mask | directly transferable | inferred | no |
| Q·K^T（ComputeMm1） | service_cube.h ComputeMm1 | `tl.dot(q, k.trans)` | rewrite | inferred | no |
| 24 模板组合分派 | template_tiling_key.h L45-86 | `if constexpr` / 独立 kernel | rewrite | inferred | no |
| PageAttention blockTable | service_cube.h KeyNd2NzForPA | `tl.load(blockTable)` → 间接 load | rewrite | inferred | no |
| core id / 四轴切分 | kernel.h SplitCore L279-357 | `tl.program_id` + `grid` | rewrite | inferred | no |
| topK 合并（sparse_count>2048） | service_vector.h ProcessLD | `tl.sort` → reduce | rewrite | inferred | no |
| CrossCore flag / MTE | - | 不迁移 | CANN-specific | inferred | no |
| sparse_count 范围 | README L60 | 需确认 >2048 场景 | needs verification | inferred | yes |
| PA_BSND 布局支持 | README L41 | 需确认 block_table 格式 | needs verification | inferred | yes |