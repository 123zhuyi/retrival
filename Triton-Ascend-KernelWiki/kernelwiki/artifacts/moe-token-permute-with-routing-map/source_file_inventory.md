# moe_token_permute_with_routing_map - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。
算子目录：`moe/moe_token_permute_with_routing_map/`。`fixture: true`。源码 ~6k 行（含 test）。

centrality：core = 直接实现该算子；supporting = 公共/框架依赖；indirect = 测试/示例/文档。SHA-256 为前 16 位。

MoE Token Permute With Routing Map 是用 routingMap（token 到 expert 映射）驱动的 MoE permute 算子。三架构（A2/A3 + 950 arch35）。核心是 maskedSelect + argSort + index_copy/gather。两种模式：dropAndPad=false/true。

## docs

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 171 | doc | core | 产品支持 L5-12；公式（2 种模式）L14-89；参数表 L91-160；约束 L162-165 | `3feb68e3521003b3…` | 算子总览 |
| `docs/aclnnMoeTokenPermuteWithRoutingMap.md` | 491 | doc/api | core | aclnn API 定义 | `6af141a3f6c28e72…` | 接口语义 |

## op_host

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_host/moe_token_permute_with_routing_map_def.cpp` | 93 | host/def | core | `MoeTokenPermuteWithRoutingMap` OpDef；输入 tokens/routing_map/probs；输出 permute_tokens/permute_probs/sorted_indices；属性 num_out_tokens（REQUIRED）L73、drop_and_pad（OPTIONAL）L74；AICore config（910b L80、910_93 L81、950 L88） | `ad76413e435fbc72…` | 算子定义 |
| `op_host/moe_token_permute_with_routing_map_tiling.cpp` | 1076 | host/tiling | core | tiling 主逻辑 | `2b8eb282e6478766…` | tiling 主逻辑 |
| `op_host/moe_token_permute_with_routing_map_tiling.h` | 120 | host/tiling | core | `MoeTokenPermuteWithRoutingMapTilingData` 结构体 | `ca06c81d12a0a765…` | tiling 结构体 |
| `op_host/op_api/aclnn_moe_token_permute_with_routing_map.cpp/.h` | 324/57 | api | core | aclnn C++ 包装 | `5a94d049608e3b5f…`/`fcf1dd85772948eb…` | aclnn API |
| `op_host/op_api/moe_token_permute_with_routing_map.cpp/.h` | 90/26 | api | core | 内部 API | `0ca484da32c24c23…`/`65752be92e72c8ac…` | 内部 API |

## op_kernel

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_kernel/moe_token_permute_with_routing_map.cpp` | 154 | kernel/entry | core | `moe_token_permute_with_routing_map` L99-154；TilingKey 分发（1/3/5/7 单核排序、2/4/6/8 多核排序、9/19 填充模式仅 3510）；调用 KernelMaskedSelectV3/MoeSort/MoeindexCopy/MoePermuteProb/Gatherv2Simd/Simt | `601cfa228160484c…` | kernel 入口 |
| `op_kernel/masked_select_v3.h` | 545 | kernel | core | `KernelMaskedSelectV3`：routingMap maskedSelect | `7e1fcb567f56b001…` | maskedSelect |
| `op_kernel/moe_sort_base.h` | 163 | kernel | core | `MoeSortBase` 排序基类 | `344e5ab8e136b99c…` | 排序基类 |
| `op_kernel/moe_sort_one_core.h` | 169 | kernel | core | `MoeSortOneCore` 单核排序 | `9a8d2cbe9b73ee26…` | 单核排序 |
| `op_kernel/moe_sort_multi_core.h` | 395 | kernel | core | `MoeSortMultiCore` 多核排序 | `d08beddd5fc473d8…` | 多核排序 |
| `op_kernel/moe_sort_multi_core_last_dim.h` | 433 | kernel | core | `MoeSortMultiLastDimCore` 末维多核排序 | `9ffbd32bdf7b5bb7…` | 末维多核排序 |
| `op_kernel/moe_mrgsort.h` | 206 | kernel | core | `MoeMrgsort` 归并排序 | `0bcc18a23f9e270a…` | 归并排序 |
| `op_kernel/moe_mrgsort_out.h` | 283 | kernel | core | `MoeMrgsortOut` 归并输出 | `39784a83b35795e1…` | 归并输出 |
| `op_kernel/moe_index_copy.h` | 257 | kernel | core | `MoeindexCopyOp` index_copy（permute tokens） | `351090612b06cf40…` | index_copy |
| `op_kernel/moe_index_copy_split_d.h` | 254 | kernel | core | `MoeindexCopySplitDOp` split-D index_copy | `6a6230a26e13785c…` | split-D index_copy |
| `op_kernel/moe_permute_prob.h` | 215 | kernel | core | `MoePermuteProb` probs 置换 | `0d688f5011f3bd90…` | probs 置换 |
| `op_kernel/moe_common.h` | 99 | kernel/common | core | 公共定义 | `1e10fcdca03af2b5…` | 公共 |
| `op_kernel/arch35/gather_v2_simd_two_dim.h` | 241 | kernel/arch35 | core | `Gatherv2SimdTwoDim` arch35 SIMD 二维 gather | `3d74314cf3cae97e…` | arch35 SIMD gather |
| `op_kernel/arch35/gather_v2_simt_two_dim.h` | 175 | kernel/arch35 | core | `Gatherv2SimtTwoDim` arch35 SIMT 二维 gather | `2882542299931f46…` | arch35 SIMT gather |

## tests

| 文件 | 行数 | role | centrality | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|
| `tests/ut/op_kernel/test_moe_token_permute_with_routing_map.cpp` | 421 | test | indirect | `65dca6f6ef476b13…` | kernel UT |
| `tests/ut/op_host/test_moe_token_permute_with_routing_map_tiling.cpp` | 133 | test | indirect | `0fa6b5fa0fb1a17e…` | tiling UT |
| `tests/ut/op_host/op_api/test_aclnn_moe_token_permute_with_routing_map.cpp` | 267 | test | indirect | `6d94d63e8c9802af…` | aclnn UT |

注：
- 核心操作：routingMap maskedSelect -> argSort -> index_copy/gather（permute tokens）+ probs 置换。
- 两种模式：dropAndPad=false（topK = numOutTokens // tokens_num）、dropAndPad=true（capacity = numOutTokens // expert_num）。
- TilingKey 1-9/19：1/3/5/7 单核排序、2/4/6/8 多核排序、9/19 填充模式（仅 3510/arch35）。
- 三架构：A2/A3（arch32）+ 950（arch35，含 SIMD/SIMT 二维 gather 专属模板）。
- 支持 BF16/FP16/FP32 tokens + BOOL/INT8 routingMap。