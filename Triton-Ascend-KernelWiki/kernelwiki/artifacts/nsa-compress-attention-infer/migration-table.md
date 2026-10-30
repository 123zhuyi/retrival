# nsa_compress_attention_infer - CANN → Triton-Ascend 迁移分类

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| NSA 压缩注意力公式 P_cmp=Softmax（QK^T）、attentionOut=P_cmp·V | docs L22-24 | `tl.dot` + online softmax | directly transferable | inferred | no |
| 滑动窗口重要性分数 P_slc = sum over 三角形权重 | .h L1362-1408 | 遍历 compressBlockSize 窗口，三角形权重累加 | directly transferable | inferred | no |
| topK 选择最重要块索引 | .h L1479-1526 | `tl.argsort` 或等效 topK | directly transferable | inferred | no |
| GQA groupSize 共享 KV | .h L901 | Q head 维循环复用同 KV | directly transferable | inferred | no |
| PagedAttention（blockTable + 3D KV） | .h L543-544, L594-596 | `tl.load（blockTable）` → 间接 `tl.load` KV | rewrite | inferred | no |
| compressBlockSize/compressBlockStride 滑窗控制 | .h L923-925 | tl.constexpr 参数 | rewrite | inferred | no |
| selectBlockSize/selectBlockCount | docs L232-250 | tl.constexpr 参数 | rewrite | inferred | no |
| core id / 任务切分（B×N2×Q-S） | tiling.cpp L327-445 | `tl.program_id` + `grid` | rewrite | inferred | no |
| 2 tiling key 分派（TND/BSND） | kernel.cpp L35-54 | `if constexpr` / 独立 kernel | rewrite | inferred | no |
| 单 token vs 多 token 路径 | .h 自动选择 | 运行时分支 | rewrite | inferred | no |
| attentionOut + topKIndices 双输出 | def.cpp L57-63 | 双返回值 | directly transferable | inferred | no |
| MTE DataCopy GM↔L1/L0/UB | copy 原语 | Triton load/store（编译器） | CANN-specific | inferred | no |
| CrossCore flag（MM1_READY/SOFTMAX_READY） | .h L34-35, L661-662 | 不迁移；抽象为 tile 依赖 | CANN-specific | inferred | no |
| 双缓冲 ping-pong flag | .h L523, L526-538 | 不迁移 | CANN-specific | inferred | no |
| MIX_AIC_1_2（Cube:Vector=1:2） | kernel.cpp L32 | 不迁移（CANN 算力配比） | CANN-specific | inferred | no |
| 滑动窗口卷积的三角形权重细节 | .h L1373-1382 | 公式 source-reported | needs verification | inferred | yes |
| tiling 参数具体数值（qSeqLenSplitSize 等） | tiling.cpp | constexpr 值 source-reported | needs verification | inferred | yes |
| UB/L1 字节预算 / workspace 占用 | tiling.cpp L452-455 | 需硬件实测 | needs verification | inferred | yes |