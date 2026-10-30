# nsa_compress_with_cache - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 |
|---|---|---|---|---|
| 压缩公式 outputCache=Σ(input×weight) | README L28-29 | `tl.sum(tl.load(input)*weight, axis=0)` | transferable | inferred |
| 压缩条件 (seqLen-compressBlockSize)%stride==0 | README L21；kernel L71-73 | `if` 条件判断 | transferable | inferred |
| slot_mapping scatter 写回 | README L23；kernel | `tl.store(cache + slot*stride)` | transferable | inferred |
| weight broadcast 乘 | kernel L80-91 | `tl.load(weight)` × 广播 | transferable | inferred |
| FP32 中间计算 | kernel L87-88 Cast to FP32 | `tl.sum(..., dtype=tl.float32)` | transferable | inferred |
| 分核（head×batch） | tiling.h coreStartBatchIdxList/HeadIdxList | `tl.program_id` + grid | rewrite | inferred |
| compressBlockSize/stride 等属性 | tiling.h | `tl.constexpr` | rewrite | inferred |
| 滑动窗口 offset 算术 | kernel InitOffset | `tl.arange` 偏移 | rewrite | inferred |
| ping-pong buffer | kernel inQueue/outQueue | 不迁移（编译器调度） | CANN-specific | inferred |
| GetBlockIdx 硬件 API | kernel L43 | 不迁移 | CANN-specific | inferred |
| tiling 参数数值 | tiling.cpp | constexpr 值 source-reported | needs verification | inferred |