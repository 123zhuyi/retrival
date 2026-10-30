# moe_init_routing - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`moe/moe_init_routing/`。`fixture: true`。
MoE（Mixture of Experts）路由算子：根据 `MoeGatingTopKSoftmax` 的输出做路由处理，对 token 特征按 expert 排序和扩展。MoE 家族：moe_init_routing、moe_init_routing_v2/v3、moe_finalize_routing、moe_fused_topk、moe_gating_top_k、moe_token_permute 等。

| 文件（相对算子目录） | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 104 | doc | core | 公式 L19-29；参数 L33-94；产品支持 L5-12 | `f8172e87257a5f71...` | 算子总览 |
| `docs/aclnnMoeInitRouting.md` | 448 | doc/api | core | 公式；参数表；约束 L98 | `9348dfb3abdab2d9...` | 接口语义 |
| `op_host/moe_init_routing_def.cpp` | 76 | host/def | core | `MoeInitRouting` OpDef L18-72；输入 x/rowIdx/expertIdx L23-40；输出 expandedX/expandedRowIdx/expandedExpertIdx L41-55；属性 active_num L56；AICore(910b/910_93/950) L63-71 | `7e87415b691a1c61...` | 算子定义 |
| `op_host/moe_init_routing_infershape.cpp` | 227 | host | core | infershape 规则 | `ef0f4122825e1d95...` | shape 推导 |
| `op_host/moe_init_routing_tiling.h` | 86 | host/tiling | core | `MoeInitRoutingTilingData` L72-83；5 个 tiling struct（VBSCompute/VMSMiddle/SortOut/GatherOut）L28-70 | `4c607ac18fca0ec5...` | tiling 结构体 |
| `op_host/moe_init_routing_tiling.cpp` | 704 | host/tiling | core | tiling 主逻辑；kernel 分派逻辑 | `a164a063aef7c2a0...` | host tiling |
| `op_kernel/moe_init_routing.cpp` | 102 | kernel/entry | core | `extern "C"` 入口 L27-101；`KERNEL_TYPE_MIX_AIV_1_0` L31；tiling key 分派 L48-99；三段流水：sort → srcToDst → gather | `9024dd0adb3971e0...` | kernel 入口 |
| `op_kernel/moe_common.h` | 82 | kernel/common | core | `MoeInitRouting` 命名空间；常量 L22-47；`Ceil`/`Align` L49-68 | `12456104d1aa3cad...` | 公共常量 |
| `op_kernel/moe_sort_multi_core.h` | 345 | kernel/sort | core | 多核排序 | `16452918308ebd97...` | 多核排序 |
| `op_kernel/moe_sort_one_core.h` | 140 | kernel/sort | core | 单核排序 | `2444e306d370135a...` | 单核排序 |
| `op_kernel/moe_sort_base.h` | 64 | kernel/sort | supporting | 排序基类 | `6d59166038b524d3...` | 排序基础 |
| `op_kernel/moe_mrgsort.h` | 207 | kernel/sort | core | merge sort 实现 | `cf48b73d5a924fa1...` | 排序 |
| `op_kernel/moe_mrgsort_out.h` | 233 | kernel/sort | core | merge sort 输出 | `9cefa29c6ef49bf7...` | 排序输出 |
| `op_kernel/moe_src_to_dst_op.h` | 183 | kernel | core | `MoeSrcToDstOp`：建立 expandedRowIdx 映射 | `096f1c863a03715f...` | 映射 |
| `op_kernel/moe_gather_out.h` | 259 | kernel | core | `MoeGatherOut`：按 expandedRowIdx gather expandedX | `19ac0dcc825fa1f6...` | gather |
| `op_kernel/moe_gather_out_small_activate_row.h` | 165 | kernel | core | 小活跃行 gather 优化 | `cdac98fb08dfcfa3...` | gather 优化 |
| `op_kernel/moe_init_routing_fullload.h` | 267 | kernel | core | `MoeFullLoad`：全量加载（tiling key 0） | `371a32a78f211dc8...` | 全量加载路径 |
| `op_kernel/moe_init_routing_apt.cpp` | 94 | kernel/apt | supporting | APT 桩 | `f03699cb37590f94...` | APT |
| `op_kernel/moe_src_to_dst_simt_op.h` | 85 | kernel/simt | supporting | SIMT 版本 srcToDst（arch35） | `189fcb9c3b12c86b...` | SIMT 映射 |
| `op_kernel/arch35/moe_common.h` | 77 | kernel/common | supporting | arch35 公共常量 | `a1384f4597ccbe53...` | arch35 |
| `op_kernel/arch35/moe_*.h` | - | kernel | supporting | arch35 各 kernel 变体 | - | arch35 专用 |
| `op_host/config/ascend***/moe_init_routing_*.json/ini` | - | config | supporting | 架构配置 | - | 架构配置 |
| `examples/test_aclnn_moe_init_routing.cpp` | 173 | test | indirect | 端到端示例 | `9f3d9f9a45ef0648...` | 示例 |
| `tests/ut/op_host/test_moe_init_routing_infershape.cpp` | 895 | test | indirect | infershape UT | `19d58ad69f46b2a3...` | 单测 |
| `tests/ut/op_kernel/test_moe_init_routing.cpp` | 297 | test | indirect | kernel UT | `009195469711d47e...` | 单测 |

注：`MIX_AIV_1_0`（纯 Vector 核，无需 Cube）。3 架构支持：ascend910b（A2）、ascend910_93（A3）、ascend950（950PR/950DT）。`arch35/` 子目录为 Ascend950 专用变体。