# moe_init_routing_quant_v2 - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（`d28d8f9...`）。arch32 + arch35 双架构。

| 文件 | 行数 | role | centrality | 关键符号/区域 | SHA-256 |
|---|---:|---|---|---|---|
| `README.md` | 222 | doc | core | 功能 L14-24；公式 L25-83；参数 L87-212 | `ef5b6f52a0eb542f...` |
| `docs/aclnnMoeInitRoutingQuantV2.md` | 630 | doc/api | core | aclnn API 语义 | `-` |
| `op_host/moe_init_routing_quant_v2_def.cpp` | 96 | host/def | core | `MoeInitRoutingQuantV2` OpDef：输入 L23-46；输出 L47-71；属性 L72-78；AICore config(910b/910_93 L84-85 + 950 L86-91) | `644fd0747e0fa64c...` |
| `op_host/moe_init_routing_quant_v2_tiling.cpp` | 602 | host/tiling | core | tiling 主逻辑 | `-` |
| `op_host/moe_init_routing_quant_v2_tiling_base.cpp` | 598 | host/tiling | core | tiling 基类 | `-` |
| `op_kernel/moe_init_routing_quant_v2.cpp` | 147 | kernel/entry | core | arch32 入口 L30-147；MIX_AIV_1_0 L43；AIV-only L44-46；3 档 tiling key(10000/20000/21000) | `5707a400460155e8...` |
| `op_kernel/moe_init_routing_quant_v2_apt.cpp` | 160 | kernel/entry | core | arch35(950) 入口 L30-160；regbase/simt 分支 | `36a5153673d015d9...` |
| `op_kernel/moe_v2_fullload_quant.h` | 518 | kernel | core | `MoeV2FullLoadQuant`（静态 quant 快速路径） | `933d68d00c35dd8c...` |
| `op_kernel/moe_v2_fullload_dynamic_quant.h` | 797 | kernel | core | `MoeV2FullLoadDynamicQuant`（动态 quant 快速路径） | `-` |
| `op_kernel/moe_v2_gather_quant.h` | 242 | kernel | core | `MoeV2GatherQuant`（静态 quant gather） | `0a848ac2515e40ce...` |
| `op_kernel/moe_v2_gather_dynamic_quant.h` | 797 | kernel | core | `MoeV2GatherDynamicQuant`（动态 quant gather） | `-` |
| `op_kernel/moe_v2_src_to_dst_and_gather.h` | 578 | kernel | core | `MoeV2SrcToDstAndGather`（融合） | `-` |
| `op_kernel/moe_v2_fullload_quant_base.h` | 284 | kernel/common | core | fullload 基类 | `-` |
| `op_kernel/inner/*.h`（13 文件） | 71-394 | kernel/common | supporting | arch32 共享：sort/src_to_dst/gather/mrgsort | `-` |
| `op_kernel/arch35/*.h`（13 文件） | 70-620 | kernel/arch35 | supporting | 950 专用：regbase/simt 变体 | `-` |

另含 `op_graph/`(1)、`tests/`(4) 等辅助文件。