# moe_token_permute - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（`d28d8f9...`）。无 arch 子目录。仅 A2/A3。

| 文件 | 行数 | role | centrality | 关键符号/区域 | SHA-256 |
|---|---:|---|---|---|---|
| `README.md` | 118 | doc | core | 功能 L14-16；公式 L18-40；参数 L42-98；约束 L100-113 | `a06fd0b15378a19d...` |
| `docs/aclnnMoeTokenPermute.md` | 419 | doc/api | core | aclnn API 语义 | `-` |
| `op_host/moe_token_permute_def.cpp` | 57 | host/def | core | `MoeTokenPermute` OpDef：输入 L23-36；输出 L37-48；属性 L49-50；AICore config(910b/910_93) L51-52 | `f759de0a2ab05eb6...` |
| `op_host/moe_token_permute_tiling.cpp` | 656 | host/tiling | core | tiling 主逻辑 | `-` |
| `op_host/moe_token_permute_tiling.h` | 93 | host/tiling | core | tiling 类 | `-` |
| `op_kernel/moe_token_permute.cpp` | 82 | kernel/entry | core | kernel 入口 L44-82；AIV-only（`g_coreType==AIC return` L47-49）；8 tiling key L65-81；3 阶段 Sort->Sort->IndexCopy | `89b44b4604988e13...` |
| `op_kernel/moe_common.h` | 72 | kernel/common | core | 公共常量 | `-` |
| `op_kernel/moe_sort_base.h` | 157 | kernel/common | core | 排序基类 | `-` |
| `op_kernel/moe_sort_one_core.h` | 167 | kernel | core | `MoeSortOneCore`（单核排序） | `-` |
| `op_kernel/moe_sort_multi_core.h` | 389 | kernel | core | `MoeSortMultiCore`（多核排序） | `-` |
| `op_kernel/moe_mrgsort.h` / `moe_mrgsort_out.h` | 207/252 | kernel | core | 归并排序 | `-` |
| `op_kernel/moe_index_copy.h` | 246 | kernel | core | `MoeindexCopyOp`（标准 index copy） | `-` |
| `op_kernel/moe_index_copy_spilt_d.h` | 248 | kernel | core | `MoeindexCopySpiltDOp`（D 维切分变体） | `-` |

另含 `op_api/`(2)、`tests/`(4) 等辅助文件。