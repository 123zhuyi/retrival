# moe_init_routing_v2 - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`。算子目录：`moe/moe_init_routing_v2/`。`fixture: true`。
MoE 路由 v2：对 v1 的增强，新增 capacity 控制、drop/pad 模式、expert_tokens 计数输出。4 架构支持（A2/A3/950/310P）。

| 文件 | 行数 | role | centrality | SHA-256 |
|---|---:|---|---|---|
| `README.md` | 172 | doc | core | `209d64a16c714e96...` |
| `docs/aclnnMoeInitRoutingV2.md` | 524 | doc/api | core | `8bf3630bc2ea6c70...` |
| `op_host/moe_init_routing_v2_def.cpp` | 151 | host/def | core | `1e687da3431b82f3...` |
| `op_host/moe_init_routing_v2_tiling.cpp` | 733 | host/tiling | core | `769a334b5429a6a0...` |
| `op_host/moe_init_routing_v2_tiling.h` | 153 | host/tiling | core | `a3bcec2641c9bd55...` |
| `op_kernel/moe_init_routing_v2.cpp` | 248 | kernel/entry | core | `122b1b6a7875178d...` |
| `op_kernel/moe_v2_common.h` | 199 | kernel/common | core | `687dc3f022ce00e9...` |
| `op_kernel/moe_v2_expert_token_out.h` | 377 | kernel | core | `a8b3960a58c28be1...` |
| `op_kernel/moe_v2_gather_out.h` | 196 | kernel | core | `4bfb030d0df1aa04...` |
| `op_kernel/moe_v2_src_to_dst_with_capacity.h` | 277 | kernel | core | `66094b6dc0d50a1e...` |
| `op_kernel/moe_v2_sort_multi_core.h` | 505 | kernel/sort | core | `3b22e6abf3b0d30a...` |

完整清单含所有 arch35 和测试文件。