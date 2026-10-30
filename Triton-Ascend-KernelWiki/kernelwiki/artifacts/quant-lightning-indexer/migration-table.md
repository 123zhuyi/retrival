# QuantLightningIndexer — 迁移分类 (S3)

依据 [references/migration-rubric.md](../../../integrations/cannbot-skills/ops/triton-kb-cann-ingest/references/migration-rubric.md) 四桶分类。所有结论基于静态源码分析（confidence: inferred）。

| 知识项 | CANN 来源 (path Lxx) | Triton-Ascend 处理 | 桶 | Confidence | Needs verification |
|--------|------|------|------|------|------|
| 量化相关性公式：out=Top_k(W@[ReLU((Scale_Q⊙Scale_K^T)⊙(Q^Quant@K^Quant^T))]) | README.md L16-20 | 直接照搬：反量化→matmul→ReLU→加权→topk | transferable | inferred | no |
| Q/K 量化输入（int8/fp8）配合 dequant_scale 反量化 | def.cpp L24-31,84-91；kernel vector.h QuantDequant | 反量化语义直接移植（`tl.load → to(tl.float32) * scale`） | transferable | inferred | no |
| per-token-head 量化模式（query_quant_mode=0, key_quant_mode=0） | def.cpp L63-64 | 语义直接移植 | transferable | inferred | no |
| ReLU 激活过滤负相关 | README.md L19 | `tl.maximum(0, x)` | transferable | inferred | no |
| topk 索引选择（sparse_count, [1,2048]） | README.md L61；kernel QLIPreload TopK | `tl.topk(x, sparse_count)` | transferable | inferred | no |
| FP32 累加（int8/fp8 输入，中间 FP32） | kernel QLIPreload | tl.dot(..).to(tl.float32) 累加 | transferable | inferred | no |
| sparse_mode=0/3（defaultMask/rightDownCausal） | README.md L63；tiling.h L81 | tl.where 因果掩码 | transferable | inferred | no |
| 单模板（QLIPreload）覆盖所有 shape | kernel L52-61 | 映射为 @triton.jit 单一 kernel | rewrite | inferred | yes（基本块数值） |
| PageAttention（PA_BSND）block_table 映射 | README.md L51；tiling.cpp GetS2SizeForPageAttention | 映射为 block_table 索引查找 + 离散 KV 加载 | rewrite | inferred | yes（PA 语义） |
| BSND/TND/PA_BSND 三 layout 分支 | tiling.h L39-43；template_tiling_key.h L26-28 | launcher 按 layout 选 stride 与 grid | rewrite | inferred | no |
| 核间切分（S1 分核，S2 不切） | tiling.cpp CalcCoreNum | grid 按 S1 分核 | rewrite | inferred | no |
| KERNEL_TYPE_MIX_AIC_1_2 | kernel L48 | 映射为 1×2 程序网格 | rewrite | inferred | no |
| MTE/DataCopy 手动搬运 | kernel QLIPreload | 抽象为 tl.load/tl.store | CANN-specific | inferred | no |
| BufferID/SetFlag/WaitFlag 同步 | kernel | 抽象为编译器排流 | CANN-specific | inferred | no |
| 手动 Cube/Vector 时序编排 | kernel Process | 仅说明意图 | CANN-specific | inferred | no |
| TBuf/TPosition UB/L1/L0C 放置 | kernel | 不移植 | CANN-specific | inferred | no |
| 基本块数值是否最优 | tiling.cpp | 源码报告值 | needs-verification | inferred | yes |
| UB 占用 | tiling.cpp | 不主张 | needs-verification | inferred | yes |
| PageAttention block_table 运行时契约 | README.md L51 | 需运行时确认 | needs-verification | inferred | yes |
| 性能（任何加速比） | - | static_ingest 无 baseline | needs-verification | inferred | yes |