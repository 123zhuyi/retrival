# moe_gating_top_k_softmax - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（`d28d8f9...`）。arch32 + arch35 双架构。

| 文件 | 行数 | role | centrality | 关键符号/区域 | SHA-256 |
|---|---:|---|---|---|---|
| `README.md` | 102 | doc | core | 功能 L16-33；公式 L19-33；参数 L37-91；约束 L93-96 | `1fced6ff88174ec3...` |
| `docs/aclnnMoeGatingTopKSoftmax.md` | 418 | doc/api | core | aclnn API 语义 | `-` |
| `op_host/moe_gating_top_k_softmax_def.cpp` | 63 | host/def | core | `MoeGatingTopKSoftmax` OpDef：输入 L23-34；输出 L35-49；属性 k L50；AICore config(910b/910_93 L51-52 + 950 L54-59) | `270d110c7a8b77da...` |
| `op_host/moe_gating_top_k_softmax_tiling.cpp` | 39 | host/tiling | core | tiling 入口 | `-` |
| `op_host/moe_gating_top_k_softmax_tiling_base.cpp` | 160 | host/tiling | core | tiling 基类 | `-` |
| `op_host/moe_gating_top_k_softmax_tiling_e_k_full_load.cpp` | 309 | host/tiling | core | E_K_FullLoad tiling | `-` |
| `op_host/moe_gating_top_k_softmax_tiling_k_full_load.cpp` | 236 | host/tiling | core | K_FullLoad tiling | `-` |
| `op_host/moe_gating_top_k_softmax_tiling_perf.cpp` | 258 | host/tiling | core | Perf tiling | `-` |
| `op_host/moe_gating_top_k_softmax_tiling_arch35.cpp` | 198 | host/tiling | core | arch35 专用 tiling | `-` |
| `op_kernel/moe_gating_top_k_softmax.cpp` | 113 | kernel/entry | core | arch32 入口 L49-113；AIV-only L53-55；18 tiling key L57-111；3 模板族 | `e00a1366c3ef28fb...` |
| `op_kernel/moe_gating_top_k_softmax_apt.cpp` | 61 | kernel/entry | core | arch35(950) 入口 | `-` |
| `op_kernel/moe_gating_top_k_softmax_e_k_fullload.h` | 473 | kernel | core | `MoeGatingTopKSoftmaxEKFullLoad`（expert×k 全装载） | `-` |
| `op_kernel/moe_gating_top_k_softmax_k_fullload.h` | 463 | kernel | core | `MoeGatingTopKSoftmaxKFullLoad`（k 全装载） | `-` |
| `op_kernel/moe_gating_top_k_softmax_perf.h` | 827 | kernel | core | `MoeGatingTopKSoftmaxPerf`（按列范围优化） | `-` |
| `op_kernel/arch35/moe_gating_top_k_softmax_fullload_generalized_regbase.h` | 533 | kernel/arch35 | core | 950 regbase fullload | `-` |

另含 `op_graph/`(1)、`framework/`(1)、`tests/`(4) 等辅助文件。