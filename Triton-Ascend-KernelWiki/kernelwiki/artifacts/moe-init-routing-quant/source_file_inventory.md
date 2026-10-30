# moe_init_routing_quant - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（`d28d8f9...`）。无 arch 子目录。仅 A2/A3。

| 文件 | 行数 | role | centrality | 关键符号/区域 | SHA-256 |
|---|---:|---|---|---|---|
| `README.md` | 117 | doc | core | 功能 L16-16；公式 L19-29；参数 L33-107 | `0b6fa60c4a82ead0...` |
| `docs/aclnnMoeInitRoutingQuant.md` | 468 | doc/api | core | aclnn API 语义 | `-` |
| `op_host/moe_init_routing_quant_def.cpp` | 65 | host/def | core | `MoeInitRoutingQuant` OpDef：输入 L23-40；输出 L41-55；属性 L56-58；AICore config(910b/910_93) L59-60 | `93cb63f9c9f686a5...` |
| `op_host/moe_init_routing_quant_tiling.cpp` | 683 | host/tiling | core | tiling 主逻辑 | `-` |
| `op_host/moe_init_routing_quant_tiling.h` | 86 | host/tiling | core | tiling 类 | `-` |
| `op_kernel/moe_init_routing_quant.cpp` | 81 | kernel/entry | core | kernel 入口 L26-81；AIV-only（`g_coreType==AIC return` L36-38）；4 tiling key L53-80；3 阶段 Sort->SrcToDst->GatherOut | `d9f4122bd40bc9ea...` |
| `op_kernel/moe_quant_common.h` | 82 | kernel/common | core | 公共常量 | `e5d9eb306059f857...` |
| `op_kernel/moe_quant_sort_multi_core.h` | 345 | kernel | core | `MoeSortMultiCore`（多核排序） | `8b56fe6439bdfd44...` |
| `op_kernel/moe_quant_sort_one_core.h` | 138 | kernel | core | `MoeSortOneCore`（单核排序） | `-` |
| `op_kernel/moe_quant_sort_base.h` | 64 | kernel/common | core | 排序基类 | `-` |
| `op_kernel/moe_quant_mrgsort.h` / `moe_quant_mrgsort_out.h` | 207/233 | kernel | core | 归并排序 | `-` |
| `op_kernel/moe_quant_src_to_dst_op.h` | 185 | kernel | core | `MoeSrcToDstOp`（反向映射 expandedRowIdx[sortedRowIdx[i]]=i） | `-` |
| `op_kernel/moe_quant_gather_out.h` | 315 | kernel | core | `MoeGatherOut`（gather + quant to INT8） | `741110eccc22860f...` |
| `op_kernel/moe_quant_gather_out_small_activate_row.h` | 231 | kernel | core | `MoeGatherOutSmallActiveRow`（小 active row 变体） | `-` |

另含 `op_graph/`(1)、`tests/`(4) 等辅助文件。