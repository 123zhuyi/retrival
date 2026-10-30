# NsaCompressAttention — 迁移分类 (S3)

依据 [references/migration-rubric.md](../../../integrations/cannbot-skills/ops/triton-kb-cann-ingest/references/migration-rubric.md) 四桶分类。所有结论基于静态源码分析（confidence: inferred）。

| 知识项 | CANN 来源 (path Lxx) | Triton-Ascend 处理 | 桶 | Confidence | Needs verification |
|--------|------|------|------|------|------|
| 压缩注意力公式：P_cmp=Softmax(QK^T)，attention_out=Softmax(mask(scale·QK^T))·V | README.md L16-22 | 直接照搬为在线 softmax matmul | transferable | inferred | no |
| importance score 滑动窗口求和 + 跨head 累加 + topk 索引 | README.md L24-38 | 滑动窗口求和 + 跨head reduce + tl.topk | transferable | inferred | no |
| compressBlockSize(l)/compressStride(d)/selectBlockSize(l')/selectBlockCount(n) 语义 | README.md L60-63 | 作为 tl.constexpr 参数 | transferable | inferred | no |
| mask/predicate（atten_mask BOOL/UINT8，1=不参与） | README.md L56；def.cpp L41-46 | tl.where 掩码 | transferable | inferred | no |
| FP32 累加（BF16/FP16 输入，中间 FP32） | tiling_general.cpp L353-354,401-414 | tl.dot(..).to(tl.float32) 累加，输出 cast 回 | transferable | inferred | no |
| 尾块/边界 mask（S1/S2 不整除基本块） | tiling.h coreParams tail 字段 | mask = offsets < n masked load/store | transferable | inferred | no |
| scale/head_num/input_layout/sparse_mode 语义 | def.cpp L92-99 | 直接作为标量参数 / tl.constexpr 分支 | transferable | inferred | no |
| 16 对齐约束、TND 仅支持、CmpSkv≤14000 | README.md L67-85 | 作为约束条件记录 | transferable | inferred | no |
| 单模板（SameAB + VarLen）覆盖全 shape | tiling_general.cpp L862-967；kernel L74-112 | 映射为 @triton.jit 单一 kernel，按 shape 选 BLOCK_* | rewrite | inferred | yes（基本块数值） |
| CV tiling 分离（C 128×S2，V 64×S2，CV_RATIO=2） | tiling_general.cpp L32,872-875 | BLOCK_M=S1 基本块，grid 按 program_id 映射 | rewrite | inferred | yes（配比数值） |
| 核间切分（按 B·N2·S1_block 分核，splitFactorSize） | tiling_general.cpp L924-942 | grid 形状 + tl.program_id | rewrite | inferred | no |
| importance score 滑动窗口 + transpose + topk 的 UB 流水 | tiling_general.cpp L796-855；kernel ProcessVec2 L580-774 | 映射为 sliding window reduce + 手工 topk | rewrite | inferred | yes（UB 参数） |
| workspace 分配（mm1Res×2 double buffer + stage1Res + impScore×2） | tiling_general.cpp L949-958；kernel L275-291 | drop（Triton 管理 temp） | rewrite | inferred | no |
| MTE/DataCopy/DataCopyPad 手动搬运 | kernel L520-533,613-646 | 抽象为 tl.load/tl.store + masked | CANN-specific | inferred | no |
| BufferID/SetFlag/WaitFlag/TQue/TBuf 同步 | kernel ProcessVec1 L469-545, ProcessVec2 L560-775 | 抽象为编译器排流 | CANN-specific | inferred | no |
| 手动 Cube/Vector 时序编排 | kernel L382-538 | 仅说明意图；Triton 由编译器排流 | CANN-specific | inferred | no |
| ConfusionTranspose ND→ND 手动执行 | kernel L653-706 | 不移植 | CANN-specific | inferred | no |
| TBuf/TPosition UB/L1/L0C 放置 | kernel L314-379 | 不移植 | CANN-specific | inferred | no |
| 基本块数值（NSA_BASE_S1G_SIZE=128 等）是否最优 | tiling_general.cpp L48；kernel L62 | 作为 BLOCK_* 源码报告值，不主张最优 | needs-verification | inferred | yes |
| UB 占用（availUbSize=192KB vs 实际） | kernel L315-321 | 不主张具体占用 | needs-verification | inferred | yes |
| CmpSkv≤14000 约束的 UB 限制 | README.md L76 | 需硬件确认 | needs-verification | inferred | yes |
| TND 布局偏移算术 | tiling_general.cpp L539-574 | 需张量实例确认 | needs-verification | inferred | yes |
| 性能（任何加速比） | - | static_ingest 无 baseline | needs-verification | inferred | yes |