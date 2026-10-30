# FusedFloydAttention — 迁移分类 (S3)

| 知识项 | CANN 来源 (path Lxx) | Triton-Ascend 处理 | 桶 | Confidence | Needs verification |
|--------|------|------|------|------|------|
| FloydAttention 公式：`weights=Softmax(attenMask+scale*(Q·K1^T+Q·K2^T))`，`out=P·V1+P·V2` | README.md L19-25 | 直接照搬为 4 次 matmul + 在线 softmax | transferable | inferred | no |
| 双 KV head group（K1/V1 + K2/V2）语义 | def.cpp L29-52 | 2 组 QK·V 分别计算后相加 | transferable | inferred | no |
| mask/predicate（atten_mask BOOL/UINT8，1=不参与） | README.md L69-72 | `tl.where` 掩码 | transferable | inferred | no |
| FP32 累加（BF16/FP16 输入，中间 FP32） | kernel 模板 | tl.dot(..).to(tl.float32) 累加 | transferable | inferred | no |
| scale_value 语义 | def.cpp L74 | `* sm_scale` | transferable | inferred | no |
| 尾块/边界 mask（N/H 不整除基本块） | tiling 尾字段 | `mask = offsets < n` | transferable | inferred | no |
| 单模板覆盖全 shape | kernel L122-148 | 映射为 @triton.jit 单一 kernel | rewrite | inferred | yes（基本块数值） |
| 4 次 matmul（Q·K1^T + Q·K2^T + P·V1 + P·V2）编排 | kernel IterateBmm1/Bmm1K1/Bmm2/Bmm2V2 | 按 CV 流水编排 | rewrite | inferred | no |
| CV tiling 分离（C/V 基本块分离） | tiling_general.cpp CalcS1S2BasicBlock | BLOCK_M/N constexpr | rewrite | inferred | yes（数值） |
| 核间切分（B/N 分核） | tiling_general.cpp SetMultiCoreParams | grid 映射 | rewrite | inferred | no |
| MTE/DataCopy 手动搬运 | kernel | `tl.load/tl.store` | CANN-specific | inferred | no |
| BufferID/SetFlag/WaitFlag 同步 | kernel | 编译器排流 | CANN-specific | inferred | no |
| TBuf/TPosition 放置 | kernel | 不移植 | CANN-specific | inferred | no |
| 基本块数值最优性 | tiling_general.cpp | 源码报告值 | needs-verification | inferred | yes |
| UB 占用 | tiling | 不主张 | needs-verification | inferred | yes |
| 性能（任何加速比） | - | static_ingest 无 baseline | needs-verification | inferred | yes |