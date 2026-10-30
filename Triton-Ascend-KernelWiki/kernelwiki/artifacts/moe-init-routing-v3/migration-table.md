# MoeInitRoutingV3 - 迁移分类 (S3)

| 知识项 | CANN 来源 | Triton-Ascend 处理 | 桶 | Confidence |
|--------|-----------|-------------------|------|------------|
| keyValueSort 排序 expertIdx | README.md 公式1 | `tl.sort` 或归并排序 | transferable | inferred |
| expandedRowIdxOut 位置映射 | README.md 公式2 | scatter 写回 `out[idx]=i` | transferable | inferred |
| Histogram 直方图（drop 模式） | README.md 公式3 | `tl.atomic_add` 或 bincount | transferable | inferred |
| 动态量化：row_max(abs(x))/127 + round | README.md 公式4 | `tl.max(abs)/127` + `tl.round` | transferable | inferred |
| 静态量化：scale/offset | README.md | `x*scale+offset` | transferable | inferred |
| expandedXOut gather | README.md 公式5 | `tl.load(quantResult[sortedRowIdx[i]%NUM_ROWS])` | transferable | inferred |
| activeExpertRange 过滤 | README.md 公式6 | mask 过滤 | transferable | inferred |
| FP32 累加（量化中间） | kernel | `tl.float32` 计算 | transferable | inferred |
| AIV-only 无 matmul | kernel L100 MIX_AIV_1_0 | 纯 sort+gather+quant | transferable | inferred |
| ~30 tiling key 多模板 | kernel L40-90 | launcher 按模式选 constexpr | rewrite | inferred |
| 单核/多核排序策略 | kernel sort_one/multi_core | 按 token 数选 grid | rewrite | inferred |
| Gather/Scatter 索引模式 | kernel | 按场景选 gather/scatter | rewrite | inferred |
| 非量化/动态/静态量化分支 | kernel full_load_* | 按 quant_mode 选 | rewrite | inferred |
| MTE/DataCopy | kernel | `tl.load/tl.store` | CANN-specific | inferred |
| 手动同步 | kernel | 编译器排流 | CANN-specific | inferred |