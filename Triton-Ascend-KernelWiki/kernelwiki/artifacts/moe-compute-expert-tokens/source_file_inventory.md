# moe_compute_expert_tokens - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（`d28d8f9...`）。无 arch 子目录。A2/A3+950+Kirin。

| 文件 | 行数 | role | centrality | 关键符号/区域 | SHA-256 |
|---|---:|---|---|---|---|
| `README.md` | 72 | doc | core | 功能 L18-23；公式 L21-23；参数 L27-60；约束 L62-66 | `a3d6274fe022835c...` |
| `docs/aclnnMoeComputeExpertTokens.md` | 375 | doc/api | core | aclnn API 语义 | `-` |
| `op_host/moe_compute_expert_tokens_def.cpp` | 53 | host/def | core | `MoeComputeExpertTokens` OpDef：输入 L23-28；输出 L29-33；属性 num_experts L34；AICore config(910b/910_93/950/kirin) L44-48 | `ca7677e88db11e5f...` |
| `op_host/moe_compute_expert_tokens_tiling.cpp` | 434 | host/tiling | core | tiling 主逻辑 | `-` |
| `op_host/moe_compute_expert_tokens_tiling.h` | 82 | host/tiling | core | tiling 类 | `-` |
| `op_kernel/moe_compute_expert_tokens.cpp` | 44 | kernel/entry | core | kernel 入口 L21-45；MIX_AIV_1_0 L24；3 tiling key(1001/1002/1003) L32-44 | `d7763c6896e46b34...` |
| `op_kernel/moe_compute_expert_tokens_int32_ss.h` | 261 | kernel | core | `MoeComputeExpertTokensInt32SS`（小数据量） | `-` |
| `op_kernel/moe_compute_expert_tokens_int32_m.h` | 452 | kernel | core | `MoeComputeExpertTokensInt32M`（中数据量） | `-` |
| `op_kernel/moe_compute_expert_tokens_int32_l.h` | 625 | kernel | core | `MoeComputeExpertTokensInt32L`（大数据量） | `-` |

另含 `op_graph/`(1)、`framework/`(1)、`tests/`(4) 等辅助文件。