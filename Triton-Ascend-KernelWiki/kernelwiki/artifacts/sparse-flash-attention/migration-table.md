# sparse_flash_attention - CANN -> Triton-Ascend 迁移分类

四分类。深度 pipeline/gather 分析见 [[technique-sparse-attention-pipeline]]、[[technique-sparse-kv-gather]]。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| 稀疏注意力公式 softmax(QK̃^T/√d)Ṽ | README L16-20 | `tl.dot` + online softmax | transferable | inferred | no |
| 在线 FlashSoftmax rescale | kernel_mla.h 流水；同 FA | running max/sum + exp rescale | transferable | inferred | no |
| MLA-absorb（nope+rope 沿 D 拼，D=576） | README L73；def.cpp attention_mode | D 维拼接 constexpr + 分段 dot | transferable | inferred | yes（拼接顺序） |
| sparse_mode rightDownCausal(3) | def.cpp L84；README L65-67 | `tl.where` 下三角 mask | transferable | inferred | no |
| scale_value 缩放 | def.cpp L80 | `s *= scale` | transferable | inferred | no |
| FP32 累加（BF16 输入） | service_vector_mla.h | `tl.dot(..., out_dtype=tl.float32)` | transferable | inferred | no |
| softmax_max/sum 输出（return_softmax_lse） | def.cpp L72-79；README L75 | 输出 running max/sum | transferable | inferred | no |
| sparse_indices 驱动离散 KV 取数 | service_vector_mla.h L821-982 | `tl.load(sparse_indices)` -> 间接 `tl.load` KV | rewrite | inferred | no |
| MergeKv 离散->连续聚合（V_TEMPLATE） | service_vector_mla.h L986 | gather 到连续 tl tensor 再 dot | rewrite | inferred | no |
| PageAttention blocktable 查表 | service_cube_mla.h L72-113 | `tl.load(block_table)` -> 间接 load | rewrite | inferred | yes（PA offset 算术） |
| mBaseSize=128 / s2BaseSize | tiling.h；tiling.cpp | `tl.constexpr BLOCK_M/N`（值 source-reported） | rewrite | inferred | yes（最优值） |
| core id / ProcessBalance 四轴 | kernel_mla.h L785-849 | `tl.program_id` + grid | rewrite | inferred | no |
| FlashDecoding S2 核间切分 | tiling.h splitKVParams | program_id 额外轴 + reduce 合并 | rewrite | inferred | yes（合并语义） |
| C_TEMPLATE/V_TEMPLATE 选择 | tiling.cpp InitParams；template_tiling_key.h | `if constexpr`/独立 kernel | rewrite | inferred | no |
| TEMPLATE_MODE 触发（sparseBlockSize≤4） | 见 technique-sparse-attention-tiling §3.1 | tiling 分支条件 | rewrite | inferred | no |
| MTE 手动 DataCopy GM↔UB（MergeKv） | service_vector_mla.h CopyInKv/CopyOut | Triton load/store（编译器） | CANN-specific | inferred | no |
| CrossCoreSetFlag/WaitFlag（5 sync event） | kernel_mla.h L88-93/738-754 | 不迁移；抽象为 tile 依赖 | CANN-specific | inferred | no |
| PreloadPipeline 3-slot 循环 buffer | kernel_mla.h L852-894 | 不迁移；抽象为 tile 复用 | CANN-specific | inferred | no |
| ND↔NZ / DataCopyPad 格式转换 | service_cube_mla.h L47-64 | Triton layout（编译器） | CANN-specific | inferred | no |
| KERNEL_TYPE_MIX_AIC_1_2（CV 1:2） | kernel entry L41 | 不迁移（CANN 算力配比） | CANN-specific | inferred | no |
| tiling 参数具体数值（s2BaseSize 等） | tiling.cpp | constexpr 值 source-reported | needs verification | inferred | yes |
| UB/L1 字节预算 | tiling.cpp CalcUbBmm | 需硬件实测 | needs verification | inferred | yes |
| sparse_block_size 与 block_size 整除约束 | README L89 | 边界，需运行时验证 | needs verification | inferred | no |
| FlashDecoding 合并数值稳定性 | kernel_mla.h | 多核 S2 切分后合并，需验证 | needs verification | inferred | yes |
