# kv_quant_sparse_flash_attention - CANN -> Triton-Ascend 迁移分类

四分类（directly transferable / rewrite to Triton-Ascend / CANN-specific do-not-transfer / needs later verification）。每条带 CANN 来源。KV Quant SFA 是 Sparse FlashAttention 的 KV 量化版本（Per-Token-Head-Tile-128 量化），仅推理场景。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| 注意力公式 Softmax(Q·Dequant(K)^T/√d)·Dequant(V) | README L14-20 | `tl.dot` + online softmax + dequant scale cast | transferable | inferred | no |
| 在线 FlashSoftmax rescale | `service_vector_mla.h` | running max/sum + `exp(m_old-m_new)` rescale | transferable | inferred | no |
| sparse_indices 驱动的离散 KV 取数 | README L42-43；`kernel_mla.h` | `tl.load(sparse_indices)` 查表 → 间接 `tl.load` KV | transferable | inferred | yes（indices 算术） |
| MLA-absorb 模式（Q/K nope+rope 拼接，D=576/656） | README L94-95；def.cpp L80-84 | Q/K 沿 D 拼接 nope+rope 分段 dot | transferable | inferred | yes（拼接顺序） |
| KV Per-Token-Head-Tile-128 量化（key_quant_mode=2 / value_quant_mode=2） | README L46-48 | dequant_scale tile 128 对齐 + cast | rewrite | inferred | yes（per-tile 反量化精确公式） |
| 量化参数和数据混合存放（quant_scale_repo_mode=1） | README L78-79 | scale 从 KV 拼接 D 维中提取 | rewrite | inferred | yes（scale 提取偏移） |
| PageAttention（block_table 驱动） | README L56-57 | `tl.load(block_table)` 查物理偏移 | rewrite | inferred | yes（PA offset 算术） |
| FP8_E4M3FN/HIFLOAT8（arch35 950） | def.cpp L103-104 | `tl.dot` FP8 输入 + FP32 累加 | rewrite | inferred | yes（FP8 精度边界） |
| sparse_block_size（1-16，2 幂次） | README L62 | `tl.constexpr` 块大小 | rewrite | inferred | no |
| 4 模板参数（FLASH_DECODE/LAYOUT_T/KV_LAYOUT_T/TEMPLATE_MODE） | `template_tiling_key.h` L31-37 | `if constexpr` / 独立 kernel | rewrite | inferred | no |
| C_TEMPLATE（Cube）vs V_TEMPLATE（Vector） | `template_tiling_key.h` L27-28 | 按 arch 选 kernel 分支 | rewrite | inferred | no |
| Tiling 分核（沿 B/N2/S1 轴） | `tiling.cpp` | `tl.program_id` + grid | rewrite | inferred | no |
| sparse_mode=3 rightDownCausal mask | README L68-71 | `tl.where` mask 逻辑 | transferable | inferred | yes（mask 精确语义） |
| MTE 离散 DataCopy GM↔UB | `service_vector_mla.h` | Triton load/store（编译器调度） | CANN-specific | inferred | no |
| Cube/Vector 手动同步 | `service_cube_mla.h`/`service_vector_mla.h` | 不迁移；抽象为 tile 复用 | CANN-specific | inferred | no |
| tiling 参数具体数值 | `tiling.cpp` | constexpr 值 source-reported | needs verification | inferred | yes |
| UB 字节预算 | `tiling.h` | 不声称；需硬件实测 | needs verification | inferred | yes |
| sparse_indices 每行有效值在前半部分 | README L42-43 | 约定而非强制 | needs verification | inferred | yes |