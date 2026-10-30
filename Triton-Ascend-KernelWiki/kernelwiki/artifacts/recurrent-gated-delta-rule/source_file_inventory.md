# recurrent_gated_delta_rule - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（`d28d8f9...`）。无 arch 子目录。

| 文件 | 行数 | role | centrality | 关键符号/区域 | SHA-256 |
|---|---:|---|---|---|---|
| `README.md` | 119 | doc | core | 功能 L16-17；RGDR 公式 L20-32；参数 L36-109；约束 L111-113 | `bb2964d80a709b79...` |
| `docs/aclnnRecurrentGatedDeltaRule.md` | 557 | doc/api | core | aclnn API 语义 | `63b27050a78c5529...` |
| `op_host/recurrent_gated_delta_rule_def.cpp` | 97 | host/def | core | `RecurrentGatedDeltaRule` OpDef：输入 L22-71；输出 L73-81；属性 L82；AICore config(910b/910_93) L84-92 | `05d617b61b70be60...` |
| `op_host/recurrent_gated_delta_rule_tiling.cpp` | 445 | host/tiling | core | tiling 主逻辑 | `f3bae2e4d74c07c6...` |
| `op_host/recurrent_gated_delta_rule_tiling.h` | 88 | host/tiling | core | `RecurrentGatedDeltaRuleTiling` 类（7 步骤：IsCapable/GetPlatformInfo/GetShapeAttrsInfo/DoOpTiling/DoLibApiTiling/GetTilingKey/GetWorkspaceSize） | `4babd14f336791d7...` |
| `op_kernel/recurrent_gated_delta_rule.cpp` | 38 | kernel/entry | core | kernel 入口 L24-38；`KERNEL_TYPE_AIV_ONLY` L31；`RGDR<bfloat16_t,bfloat16_t>` L33 | `5ea0cecbe238a02e...` |
| `op_kernel/recurrent_gated_delta_rule.h` | 429 | kernel | core | `RGDR` 类：Init L72+（SetGlobalTensors/CalcUbSize/分块）；Process（循环 S_t 更新 + o_t 输出） | `3681292c16a289e6...` |
| `op_kernel/recurrent_gated_delta_rule_tiling_data.h` | 42 | kernel | core | `RecurrentGatedDeltaRuleTilingData`（vectorCoreNum/ubCalSize/ubRestBytes/t/nk/dk/nv/dv/sBlockNum/b/vStep/scale/hasGama/hasGamaK/hasAcceptedTokens） | `b55f5cf11d5b3d94...` |

另含 `op_api/`(4)、`tests/ut/`(4) 等辅助文件。