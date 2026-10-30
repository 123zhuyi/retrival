# moe_token_permute_with_ep - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（`d28d8f9...`）。无 arch 子目录。仅 A2/A3。

| 文件 | 行数 | role | centrality | 关键符号/区域 |
|---|---:|---|---|---|
| `README.md` | 140 | doc | core | 功能 L16-48；参数 L52-126；约束 L128-133 |
| `docs/aclnnMoeTokenPermuteWithEp.md` | - | doc/api | core | aclnn API 语义 |
| `op_host/moe_token_permute_with_ep_def.cpp` | 72 | host/def | core | `MoeTokenPermuteWithEp` OpDef：输入 L23-43；输出 L44-61；属性 L62-64；AICore config(910b/910_93) L66-67 |
| `op_host/moe_token_permute_with_ep_tiling.cpp` | - | host/tiling | core | tiling 主逻辑 |
| `op_host/moe_token_permute_with_ep_tiling.h` | - | host/tiling | core | tiling 类 |
| `op_host/moe_token_permute_tiling_base.h` | - | host/tiling | core | 共享 tiling 基类（与 token_permute 共用） |
| `op_kernel/moe_token_permute_with_ep.cpp` | 86 | kernel/entry | core | kernel 入口 L47-86；AIV-only（`g_coreType==AIC return` L51-53）；8 tiling key L69-85；3 阶段 Sort->Sort->IndexCopy（含 probs） |
| `op_kernel/moe_common.h` | 72 | kernel/common | core | 公共常量（与 token_permute 共用） |
| `op_kernel/moe_sort_base.h` | 157 | kernel/common | core | 排序基类 |
| `op_kernel/moe_sort_one_core.h` | 167 | kernel | core | `MoeSortOneCore` |
| `op_kernel/moe_sort_multi_core.h` | 389 | kernel | core | `MoeSortMultiCore` |
| `op_kernel/moe_mrgsort.h` / `moe_mrgsort_out.h` | 207/252 | kernel | core | 归并排序 |
| `op_kernel/moe_index_copy.h` | 246 | kernel | core | `MoeindexCopyOp`（含 probs copy） |
| `op_kernel/moe_index_copy_spilt_d.h` | 248 | kernel | core | `MoeindexCopySpiltDOp`（D 维切分 + probs） |

注：kernel 文件与 `moe_token_permute` 高度同构，差异在 index_copy 同时处理 tokens + probs，且按 range 切片。