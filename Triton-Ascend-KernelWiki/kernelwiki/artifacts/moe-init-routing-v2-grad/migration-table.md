# MoeInitRoutingV2Grad - 迁移分类 (S3)

| 知识项 | CANN 来源 | Triton-Ascend 处理 | 桶 | Confidence |
|--------|-----------|-------------------|------|------------|
| 反向 gather + reduce sum 公式 | README.md | `gradX_i = Σ_t gradExpandedX[expandedRowIdx[i*topK+t]]` 直接照搬 | transferable | inferred |
| expandedRowIdx 索引映射 | README.md | `tl.load(gradExpandedX + idx * stride)` | transferable | inferred |
| FP32 累加 | kernel 模板 | `tl.load(..).to(tl.float32)` 累加 | transferable | inferred |
| AIV-only 无 matmul | kernel L39 | 纯 gather + reduce | transferable | inferred |
| top_k/drop_pad_mode/active_num 语义 | def.cpp L39-41 | 作为标量参数 | transferable | inferred |
| 3 模式（Dropless/Activate/PositionPad0） | kernel L47-83 | launcher 按模式选 constexpr | rewrite | inferred |
| arch35 三策略（SplitH/FullLoad/通用） | tiling.h L52-93 | 按 H 大小选 BLOCK_H | rewrite | inferred |
| 核间切分（needCoreNum/perCoreElements） | tiling.h L22-24 | grid 映射 | rewrite | inferred |
| MTE/DataCopy | kernel | `tl.load/tl.store` | CANN-specific | inferred |
| 手动同步 | kernel | 编译器排流 | CANN-specific | inferred |