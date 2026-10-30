# MoeTokenUnpermuteWithEp - 迁移分类 (S3)

| 知识项 | CANN 来源 | Triton-Ascend 处理 | 桶 | Confidence |
|--------|-----------|-------------------|------|------------|
| range 过滤 sortedIndices | README.md 公式1 | mask 过滤 | transferable | inferred |
| indexSelect gather | README.md 公式2 | `tl.load(permutedTokens + sortedIndices[i] * stride)` | transferable | inferred |
| 乘 probs（probs 非 None） | README.md 公式3 | `* probs[i,j]` | transferable | inferred |
| 跨 topK reduce sum | README.md 公式4 | `tl.sum` 跨 topK 维 | transferable | inferred |
| FP32 累加 | kernel | `tl.float32` 累加 | transferable | inferred |
| AIV-only 无 matmul | kernel | 纯 gather + reduce | transferable | inferred |
| num_topk/range/padded_mode/restore_shape 语义 | def.cpp L56-59 | 作为标量参数 | transferable | inferred |
| 4 tiling key（无 probs/float/half/bf16 probs） | kernel L39-49 | launcher 按 probs dtype 选 constexpr | rewrite | inferred |
| 核间切分 | tiling | grid 映射 | rewrite | inferred |
| MTE/DataCopy | kernel | `tl.load/tl.store` | CANN-specific | inferred |
| 手动同步 | kernel | 编译器排流 | CANN-specific | inferred |