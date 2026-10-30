# MoeFinalizeRoutingV2 — 迁移分类 (S3)

| 知识项 | CANN 来源 | Triton-Ascend 处理 | 桶 | Confidence |
|--------|-----------|-------------------|------|------------|
| gather + scale + bias + reduce sum 公式 | README.md L22-27 | 直接照搬：`gather(expandedX, idx) * scales + bias` | transferable | inferred |
| expandedRowIdx 索引映射 | README.md | `tl.load(expandedX + idx * stride)` | transferable | inferred |
| FP32 累加（BF16/FP16 输入，中间 FP32） | kernel 模板 | `tl.load(..).to(tl.float32)` | transferable | inferred |
| AIV-only 无 matmul | kernel L85 KERNEL_TYPE_AIV_ONLY | 纯 elementwise 操作 | transferable | inferred |
| 多模板策略（CutK/Db/Cuth/CuthK2/CuthK4） | kernel L92-298 | 映射为 launcher 按 H/K 选 BLOCK_H/BLOCK_K | rewrite | inferred |
| H/K 轴切分（hSliceNum/kSliceNum） | tiling.h L86-89 | grid 映射 | rewrite | inferred |
| 核间/核内切分（normalCore/tailCore） | tiling.h L91-98 | program_id 映射 | rewrite | inferred |
| MTE/DataCopy | kernel | `tl.load/tl.store` | CANN-specific | inferred |
| 手动同步 | kernel | 编译器排流 | CANN-specific | inferred |