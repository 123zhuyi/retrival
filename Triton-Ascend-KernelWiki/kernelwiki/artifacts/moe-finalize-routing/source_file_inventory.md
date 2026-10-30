# moe_finalize_routing - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（`d28d8f9...`）。无 arch 子目录。A2/A3 + Kirin。

| 文件 | 行数 | role | centrality | 关键符号/区域 | SHA-256 |
|---|---:|---|---|---|---|
| `README.md` | 109 | doc | core | 功能 L18-19；公式 L20-27；参数 L29-99 | `4b4a77711d30e868...` |
| `docs/aclnnMoeFinalizeRouting.md` | 457 | doc/api | core | aclnn API 语义 | `5343b73324039c24...` |
| `op_host/moe_finalize_routing_def.cpp` | 148 | host/def | core | `MoeFinalizeRouting` OpDef：输入 L22-63；输出 L64-68；AICore config(910b/910_93 L79-80 + Kirin L82-84) | `fca675ec1e6ed1ab...` |
| `op_host/moe_finalize_routing_tiling.cpp` | 630 | host/tiling | core | tiling 主逻辑 | `a74c754733775c4e...` |
| `op_host/moe_finalize_routing_tiling.h` | 125 | host/tiling | core | tiling 类 | `df8aca834709247c...` |
| `op_kernel/moe_finalize_routing.cpp` | 143 | kernel/entry | core | 19 tiling key(20000-20018) 分支 L43-142；`KernelFinalizeRouting` 模板族 | `efe91ee712995142...` |
| `op_kernel/moe_finalize_routing_common.h` | 39 | kernel/common | core | `BUFFER_NUM=1`, `PARALLEL_NUM=2`, `ONE_BLK_SIZE=32` L23-27 | `860044e030eecdae...` |
| `op_kernel/moe_finalize_routing_fp.h` | 365 | kernel | core | FP 基准模板 | `4f432e480d6fdb5b...` |
| `op_kernel/moe_finalize_routing_fp_cuth.h` | 457 | kernel | core | FP Cuth 变体 | `1a9de8d16e587922...` |
| `op_kernel/moe_finalize_routing_fp_cuth_k2.h` | 405 | kernel | core | FP Cuth K2 | `3a2b1e328e27d495...` |
| `op_kernel/moe_finalize_routing_fp_cuth_k2_optimized.h` | 495 | kernel | core | FP Cuth K2 Optimized | `301810ba05e7db84...` |
| `op_kernel/moe_finalize_routing_fp_cuth_k4.h` | 441 | kernel | core | FP Cuth K4 | `88dcf7ca90b91a6e...` |
| `op_kernel/moe_finalize_routing_fp_cutk.h` | 416 | kernel | core | FP CutK | `79fe74c3267fb7d4...` |
| `op_kernel/moe_finalize_routing_fp_db.h` | 437 | kernel | core | FP Db（double buffer） | `1ba1cd887e7475a6...` |
| `op_kernel/moe_finalize_routing_fp_db_all_bias.h` | 467 | kernel | core | FP Db AllBias | `6f7931130f5276de...` |
| `op_kernel/moe_finalize_routing_bf16.h` | 469 | kernel | core | BF16 基准模板 | `fb25ed70eff8a134...` |
| `op_kernel/moe_finalize_routing_bf16_all_bias.h` | 506 | kernel | core | BF16 AllBias | `8a92f4ce923805fc...` |
| `op_kernel/moe_finalize_routing_bf16_cuth.h` | 489 | kernel | core | BF16 Cuth | `495f72d524d5a63a...` |
| `op_kernel/moe_finalize_routing_bf16_cuth_k2.h` | 436 | kernel | core | BF16 Cuth K2 | `0098852ec6fb3627...` |
| `op_kernel/moe_finalize_routing_bf16_cuth_k4.h` | 494 | kernel | core | BF16 Cuth K4 | `d19137d25265f5e4...` |
| `op_kernel/moe_finalize_routing_bf16_cutk.h` | 441 | kernel | core | BF16 CutK | `1f4ca05cacdf1f89...` |

另含 `op_graph/`(1)、`framework/`(1)、`tests/`(4) 等辅助文件。