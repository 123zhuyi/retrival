# fused_infer_attention_score - CANN -> Triton-Ascend 迁移分类

四分类（directly transferable / rewrite to Triton-Ascend / CANN-specific do-not-transfer / needs later verification）。每条带 CANN 来源。FIA 为推理统一 attention（prefill+decode），核心架构是 `attn_infra/` 分层模板库。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| 注意力公式 Softmax(QK^T/√d)V | README L24-34 | `tl.dot`+online softmax | transferable | inferred | no |
| 在线 FlashSoftmax rescale | `attn_infra/epilogue/block/block_epilogue_online_softmax.hpp` | running max/sum + `exp(m_old-m_new)` rescale PV 累加 | transferable | inferred | yes（精确实现以源码为准） |
| 统一 prefill（Q_S>1）+ decode（Q_S=1） | README L16；V5 doc L19 | `BLOCK_M` 由 Q_S 决定（prefill 多行 / decode=1） | transferable | inferred | no |
| mask select->-inf（atten_mask） | V5 doc L189-210 | `tl.where(mask, -inf, s)` | transferable | inferred | no |
| pse（pseType，V5 新增） | V5 doc L21/属性 pseType | `s = s*pse or s+pse`（按 pseType） | transferable | inferred | yes（pseType 取值语义） |
| scale_value 缩放 | V5 doc 属性 scaleValue | `s *= sm_scale` | transferable | inferred | no |
| FP32 累加（BF16/FP8/INT8 输入） | attn_infra gemm FP32 出 | `tl.dot(..., out_dtype=tl.float32)` | transferable | inferred | no |
| KV cache（decode 历史累积） | V5 doc L25 | actualSeqLengthsKv 驱动 per-batch KV 长度 | transferable | inferred | no |
| 变长 actualSeqLengths / actualSeqLengthsKv | V5 doc L211-240 | per-batch offset + mask | transferable | inferred | yes（TND packing offset） |
| FlashDecode（split-KV lse 规约） | `tiling.cpp` L1426-1439；`flash_attention_regular_decode.h` | 两段式（split + lse 加权 combine） | transferable | inferred | yes（lse 精确公式） |
| softmaxLse 输出（softmaxLseFlag） | V5 doc 属性 softmaxLseFlag | `lse = log(l_i) + m_i` 输出 | transferable | inferred | yes（精确公式） |
| query/kv padding（queryPaddingSize/kvPaddingSize） | V5 doc L363-393 | 起始偏移 = `Smax - padding - actualLen`，越界 0 | transferable | inferred | yes（与 mask 联动精度） |
| qStartIdx/kvStartIdx（V5 新增） | V5 doc L21/85-86 | Q/KV 序列起始偏移 | transferable | inferred | yes（offset 算术） |
| sharedPrefix（共享前缀） | V5 doc L464-499；`tiling.cpp` | 前缀 KV 独立 load + 拼接 | rewrite | inferred | yes（前缀拼接顺序） |
| RoPE 解耦（queryRope/keyRope） | V5 doc L80-81 | Q/K 沿 D 拼接 nope+rope 分段 dot | rewrite | inferred | yes（拼接顺序） |
| learnableSink | V5 doc L84 | sink 项加入 attention 分母 | rewrite | inferred | yes（精确公式） |
| 分离 KV 反量化（keyAntiquantScale/valueAntiquantScale，V5） | V5 doc L394-463 | K/V 各自 cast + scale*offset 融合 | rewrite | inferred | yes（per-token/per-head 模式） |
| queryQuantMode | V5 doc 属性 queryQuantMode | Q 量化模式分支 | rewrite | inferred | yes（模式枚举） |
| 量化 deq_scale1/2 + quant_scale1/2 | V5 doc L241-327 | cast + scale 融合（per-tensor/channel） | rewrite | inferred | yes（精度边界） |
| PageAttention（blockTable） | V5 doc L348-362 | `tl.load(block_table)` 查物理偏移 -> 间接 load | rewrite | inferred | yes（PA offset 算术） |
| MLA | `tiling.cpp` L250-263 | D 维拼接 constexpr + 分段 dot | rewrite | inferred | yes（IFA/FIA MLA 范围） |
| **attn_infra 分层（gemm/epilogue/layout/arch）** | `op_kernel/attn_infra/` 全部 | Triton tile 分层：`tl.dot`=gemm，online softmax=epilogue，layout=编译器，arch=buffer 管理 | rewrite | inferred | yes（精确类名/行号） |
| gemm block_mmad_qk/pv（Bmm1/Bmm2） | `attn_infra/gemm/block/block_mmad_qk.hpp`/`block_mmad_pv.hpp` | `tl.dot(q,k.T)` / `tl.dot(p,v)` | rewrite | inferred | no |
| epilogue online_softmax/rescale_o | `attn_infra/epilogue/block/block_epilogue_online_softmax.hpp`/`block_epilogue_rescale_o.hpp` | running softmax + `o=acc/l_i` | rewrite | inferred | no |
| tilingkey 十进制组装 | `tilingkey.h`；`tiling.cpp` GenTilingKey L923-924 | `if constexpr`/launcher dispatch（key 不迁移） | rewrite | inferred | no |
| 模板分发（regular/regular_decode + attn_infra） | `flash_attention_regular.h`/`_regular_decode.h`；`_template_tiling_key.h`（3116行） | 按 Q_S/dtype/feature 选 `@triton.jit` kernel | rewrite | inferred | no |
| arch32 vs arch35 模板 | `def.cpp` L568-571/L2048-2050 | 按 arch 选 BLOCK_* / attn_infra 分层 | rewrite | inferred | no |
| layout ND↔NZ | `attn_infra/layout/matrix.hpp` | Triton layout（编译器管理） | CANN-specific | inferred | no |
| arch buffer/sync/cross_core_sync | `attn_infra/arch/*` | 不迁移；Triton 编译器管理 buffer/sync | CANN-specific | inferred | no |
| MTE 手动 DataCopy GM↔UB/L1 | `attn_infra/gemm/tile_common/copy_*.hpp` | Triton load/store（编译器调度） | CANN-specific | inferred | no |
| BufferID/queue/event/CV 手动同步 | `attn_infra/arch/cross_core_sync.hpp` | 不迁移；抽象为 tile 复用 | CANN-specific | inferred | no |
| Cube L1/L0A/L0B/L0C 物理搬运 | `copy_l1_to_l0a/l0b`、`copy_l0c_to_gm` | 不迁移；Triton 编译器管理 | CANN-specific | inferred | no |
| tiling 参数具体数值 | `tiling.cpp`/`tiling_v3.cpp`/`tiling_v2.cpp` | constexpr 值 source-reported，不声称最优 | needs verification | inferred | yes |
| UB/L1/L0C 字节预算 | `tiling.cpp` workspace L1536-1537 | 不声称；需硬件实测 | needs verification | inferred | yes |
| V5 仅 950 支持 | V5 doc L10-15（× A2/A3） | V5 接口在 A2/A3 支持范围 | needs verification | inferred | yes |
| FP8/FP4/INT4 精度边界 | V5 doc dtype（HIFLOAT8/FLOAT8_E4M3FN/INT4/FLOAT4_E2M1） | 需运行时验证 | needs verification | inferred | yes |
| attn_infra 精确类名/行号 | `attn_infra/*.hpp` | 关键符号由文件名推断，以源码为准 | needs verification | inferred | yes |
| `tiling.cpp` 关键逻辑精确行号 | `tiling.cpp` L100-1548 | 行号静态读取，可能偏移 | needs verification | inferred | yes |
| 长序列超时阈值 | README L100-154（S≤20M） | 依赖 B/S/N/D 乘积，无固定值 | needs verification | inferred | yes |
