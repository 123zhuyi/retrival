# block_sparse_attention_grad - CANN -> Triton-Ascend 迁移分类

四分类。前向迁移表见 `artifacts/block-sparse-attention/migration-table.md`；块稀疏 mask 方法见 [[technique-block-sparse-mask-gather]]（前向 CSR）；反向 mask 门控扫描见 [[kernel-block-sparse-attention-grad]] §3。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| 反向公式 dP/dS/dV/dQ/dK | docs L19-71；README L19-71 | `tl.dot` 组合 | directly transferable | inferred | no |
| softmax-grad D=Σ(dY·O) | block_epilogue_softmaxgrad.hpp L208-297 | `tl.sum(dy*o, axis=-1)` | directly transferable | inferred | no |
| dS=P·(dP−D) | block_epilogue_simply_softmax.hpp L402-444 | `p*(dp - D[:,None])` | directly transferable | inferred | no |
| 重算 P=exp(S·scale−Lse)（用前向 Lse） | block_epilogue_simply_softmax.hpp L335-393 | `p=tl.exp(tl.dot(q,k)*scale - tl.load(lse))` | directly transferable | inferred | no |
| scale_value（dQ/dK 乘 scale，dV 不） | block_epilogue_post.hpp L371-374 | `dq*=scale; dk*=scale`（dV 不乘） | directly transferable | inferred | yes（dV 是否乘 scale） |
| FP32 累加（dQ/dK/dV） | tiling.cpp L432-434；block_mmad_bsag2 | `tl.dot(..., out_dtype=tl.float32)` | directly transferable | inferred | no |
| GQA groupSize 共享 KV | tiling.cpp L217-221；kernel.h L302 | Q head 维循环复用 KV | directly transferable | inferred | no |
| blockSparseMask 语义（块级 1/0） | docs L213-216 | mask 决定 KV 块参与 | directly transferable | inferred | no |
| ceilDiv 非对齐 + tail | tiling.cpp L80-87 | `tl.arange` + `mask=` | directly transferable | inferred | no |
| **mask 门控稠密扫描**（遍历所有 KV 块，`if(mask) skip`） | kernel.h L386-389,604-607 | `m=tl.load(mask_row); tl.where(m, s, -inf)` 或 `if not m: continue` | rewrite | inferred | yes（vs CSR gather 的取舍） |
| 重算 P（非在线 softmax） | SimpltSoftmax L335-393 | 显式重算 P 块 | rewrite | inferred | no |
| 3 cube GEMM（S&dP / dQ / dK&dV） | kernel.h L405-460；block_mmad_bsag2 | `tl.dot` 三组 | rewrite | inferred | no |
| dQ/dK/dV AtomicAdd 跨 KV 块累加 | block_mmad_bsag2 L159-162（Cube2/3 isAtomicAdd=true） | `tl.atomic_add` 或 per-program FP32 accum | rewrite | inferred | yes（atomic vs 拆核 reduce） |
| core id / 任务切分（Q-block×head） | tiling.cpp L369-420；kernel.h L216-221 | `tl.program_id` + `grid` | rewrite | inferred | no |
| 4 tiling key 分派 | block_sparse_attention_grad.cpp L44-60 | `if constexpr` / 独立 kernel | rewrite | inferred | no |
| 4 阶段 cube↔vec 流水 | kernel.h L291-531 | 抽象为 tile 循环 + fused op | rewrite | inferred | no |
| BASIC_BLOCK_SIZE=128 / blockShape | tiling.cpp L68 | `tl.constexpr`（值 source-reported） | rewrite | inferred | yes（最优值） |
| MTE DataCopy GM↔L1/L0/UB | copy_*.hpp；kernel.h L341-355 | Triton load/store（编译器） | CANN-specific | inferred | no |
| CrossCore flag（CUBE2VEC/VEC2CUBE/CUBE2POST, id 7/8/9） | kernel.h L39-41,422,475,511 | 不迁移；抽象为 tile 依赖 | CANN-specific | inferred | no |
| preTaskInfo 1-deep delay（dQ/dK/dV 晚一拍） | kernel.h L423-461 | 不迁移；抽象为 tile 复用 | CANN-specific | inferred | no |
| TQue/LocalTensor/BufferID | local_tensor_buffer.hpp | 不迁移 | CANN-specific | inferred | no |
| ND↔NZ / DataCopyPad | layout/matrix.hpp | 不迁移；Triton layout | CANN-specific | inferred | no |
| AtomicAdd（CANN 硬件原语） | block_mmad_bsag2 L159-162 | `tl.atomic_add`（语义等价） | CANN-specific | inferred | no |
| SoftmaxGradFront/SoftMaxGradTilingFunc | block_epilogue_softmaxgrad.hpp L280；tiling.cpp L296 | 不迁移；手写 reduction | CANN-specific | inferred | no |
| tiling 参数具体数值（postUbBaseSize 等） | tiling.cpp | constexpr 值 source-reported | needs verification | inferred | yes |
| UB/L1 字节预算 / workspace 占用 | tiling.cpp L422-447 | 需硬件实测 | needs verification | inferred | yes |
| mask dtype（def=UINT8 / README=BOOL / kernel=bool） | def.cpp L43-46；README L178；kernel.h L334 | 三处不一致，需确认 | needs verification | inferred | yes |
| headdim=128 强制 | README L366；tiling.cpp L61(定义但未强制) | 文档说仅 128，代码未强制 | needs verification | inferred | yes |
| TND+GQA 支持 | README L367（仅 BNSD+MHA）vs 参数表 L84-85 | 文档/代码不一致 | needs verification | inferred | yes |
| 任务类型宏缺失 | block_sparse_attention_grad.cpp（无 MIX_AIC） | 是否 host/tiling 设置待确认 | needs verification | inferred | yes |
| scaleValue=0.0 默认 | def.cpp L80；tiling.cpp L198-202 | null 才取 1/√D，0.0 是否归一化 | needs verification | inferred | yes |
| SBSAG1/SBSAG3 声明但无特化 | gemm/dispatch_policy.hpp L72-82 | 仅 SBSAG2 有 BlockMmad，余未用 | needs verification | inferred | yes |
