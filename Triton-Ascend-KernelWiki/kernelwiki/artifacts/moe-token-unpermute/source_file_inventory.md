# moe_token_unpermute - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（`d28d8f9...`）。无 arch 子目录。A2/A3 + 310P + Kirin。

| 文件 | 行数 | role | centrality | 关键符号/区域 | SHA-256 |
|---|---:|---|---|---|---|
| `README.md` | 125 | doc | core | 功能 L18-47；公式 L21-47；参数 L49-104；约束 L106-119 | `dc9e84e42d2251d4...` |
| `docs/aclnnMoeTokenUnpermute.md` | 483 | doc/api | core | aclnn API 语义 | `-` |
| `op_host/moe_token_unpermute_def.cpp` | 123 | host/def | core | `MoeTokenUnpermute` OpDef：输入 L23-58；输出 L63-73；属性 L60-61；AICore config(910b/910_93 L75-76 + 310p/kirin L78-118) | `abdc50111f6158a4...` |
| `op_host/moe_token_unpermute_tiling.cpp` | 379 | host/tiling | core | tiling 主逻辑 | `-` |
| `op_host/moe_token_unpermute_tiling.h` | 99 | host/tiling | core | tiling 类 | `-` |
| `op_kernel/moe_token_unpermute.cpp` | 88 | kernel/entry | core | kernel 入口 L27-88；~16 tiling key（dtype×haveProbs×mix） L33-87；`KernelMoeTokenUnpermute` 模板 | `2e2de67a11696724...` |
| `op_kernel/moe_token_unpermute.h` | 376 | kernel | core | `KernelMoeTokenUnpermute`：Process（gather + probs mul + reduce_sum） | `-` |
| `op_kernel/moe_token_unpermute_common.h` | 85 | kernel/common | core | 公共常量 | `-` |

另含 `op_api/`(2)、`tests/`(5) 等辅助文件。