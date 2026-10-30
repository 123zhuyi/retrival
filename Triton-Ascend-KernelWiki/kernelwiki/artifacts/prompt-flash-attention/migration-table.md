# prompt_flash_attention — CANN → Triton-Ascend 迁移分类

四分类（rubric：directly transferable / rewrite to Triton-Ascend / CANN-specific do-not-transfer / needs later verification）。每条带 CANN 来源。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| 注意力公式 Softmax(QK^T/√d)V | README L24-32 | `tl.dot`+`tl.softmax`/online | transferable | inferred | no |
| 在线 FlashSoftmax rescale | PFA设计介绍 L13-22 | running max/sum + `exp(m_old-m_new)` rescale PV 累加 | transferable | inferred | no |
| mask select→-inf（atten_mask true 遮蔽） | PFA设计介绍 L11 | `tl.where(mask, -inf, s)` | transferable | inferred | no |
| pse add/mul 顺序（pseType） | def.cpp 属性；enum PseMode L189-194 | `if pse_mode: s = s*pse or s+pse` | transferable | inferred | no |
| scale_value 缩放 | def.cpp L83-85 | `s *= sm_scale` | transferable | inferred | no |
| FP32 累加（BF16/FP8 输入） | PFA设计介绍 L59（FP32 中间） | `tl.dot(..., out_dtype=tl.float32)` | transferable | inferred | no |
| softmaxLse 输出（推理） | prompt_flash_attention.cpp L36 | `lse = log(l_i) + m_i` 输出 | transferable | inferred | yes（精确公式以源码为准） |
| 空输入/空 KV 处理 | README L92；tiling.cpp L73/350 | early-return / 全零 / mask 全 False | transferable | inferred | no |
| 变长 actualSeqLengths（TND） | def.cpp L46-54；register.cpp L55 | per-batch offset + mask | transferable | inferred | yes（TND packing offset） |
| S1/S2 基本块 128/1024 | tiling.cpp L93-106 | `tl.constexpr BLOCK_M=128, BLOCK_N=1024`（值 source-reported） | rewrite | inferred | yes（最优值依赖 shape） |
| core id / splitCoreMode | tiling.cpp L926-932 | `tl.program_id` + grid 轴 | rewrite | inferred | no |
| MLA D=576(nope+rope) | enum D L65；tiling.cpp L2848 | D 维拼接 constexpr + 分段 dot | rewrite | inferred | yes（拼接顺序） |
| tilingkey 十进制组装 | tilingkey.h；PFA设计介绍 L61-92 | `if constexpr`/launcher dispatch（key 本身不迁移） | rewrite | inferred | no |
| arch32 vs arch35 模板 | def.cpp L114-127 | 按 arch 选 BLOCK_*；单/多 kernel | rewrite | inferred | no |
| PagedAttention blockTable | tiling.cpp L1959/1987 | `tl.load(block_table)` 查表 → 间接 load | rewrite | inferred | yes（PA offset 算术） |
| 量化 deq_scale/quant_scale/antiquant | def.cpp L56-75；enum QuantMode L197-206 | cast + scale 融合（per-channel/token） | rewrite | inferred | yes（MSD/antiquant 精度边界） |
| MTE 手动 DataCopy GM↔UB | PFA设计介绍 L55-59 | Triton load/store（编译器调度） | CANN-specific | inferred | no |
| BufferID/queue/event/CV 手动同步 | PFA设计介绍 L119-138 | 不迁移；抽象为 tile 复用/fused load-store | CANN-specific | inferred | no |
| ND↔NZ 物理格式转换 | arch35 copy_cube_in/matmul_modules | Triton layout（编译器管理） | CANN-specific | inferred | no |
| Cube 双发 / nRatio=8 CV 配比 | PFA设计介绍 L59/121 | 不迁移（CANN 算力配比） | CANN-specific | inferred | no |
| TBuf 放置 UB/L1/L0 + 平台查询 | register.cpp L34-42 | 不迁移；launcher 按 shape 选 meta | CANN-specific | inferred | no |
| tiling 参数具体数值（S1.i=128 最优等） | tiling.cpp/tiling_const.h | constexpr 值 source-reported，不声称最优 | needs verification | inferred | yes |
| UB/L1/L0C 字节预算/占用 | tiling.cpp L465-490 | 不声称；需硬件实测 | needs verification | inferred | yes |
| FP8/FP4 精度边界 | tilingkey.h QFP8/QFP4 | 需运行时验证 | needs verification | inferred | yes |
| softmaxLse 精确公式 | vf_softmaxflashv3.h | 以源码为准 | needs verification | inferred | yes |
| KV-cache flashupdate 写回时机 | vf_flashupdate.h | prefill 是否同时写 cache 待确认 | needs verification | inferred | yes |
| 长序列超时阈值 | README L102 | 依赖 B/S/N/D 乘积，无固定值 | needs verification | inferred | yes |
