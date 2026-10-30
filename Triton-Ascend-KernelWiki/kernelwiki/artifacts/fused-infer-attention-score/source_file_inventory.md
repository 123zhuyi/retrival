# fused_infer_attention_score - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/fused_infer_attention_score/`。`fixture: true`。源码 ~39k 行（+ `attn_infra/` 52 个 `.hpp` ~14k 行）。

centrality：core = 直接实现该算子；supporting = 公共/框架依赖；indirect = 测试/示例/文档。SHA-256 为前 16 位。

FIA 是推理统一 attention（prefill Q_S>1 + decode Q_S=1），架构分 arch32（A2/A3）与 arch35（950）；op_kernel 的 `attn_infra/` 是 header-only C++ 模板库，按 gemm/epilogue/layout/arch 分层，是 FIA 区别于 PFA/IFA 的核心架构。

## docs

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 173 | doc | core | 产品支持 L5-12；功能（统一 prefill+decode）L14-22；参数 L36-83；约束 L85-167 | `0d06ea50c9f93df6…` | 算子总览 |
| `docs/aclnnFusedInferAttentionScore.md` | 1084 | doc/api | indirect | V1 aclnn API | `3b83f2f59fc30dbc…` | V1 接口 |
| `docs/aclnnFusedInferAttentionScoreV2.md` | 1411 | doc/api | core | V2 API | `7a274600746f680e…` | V2 接口 |
| `docs/aclnnFusedInferAttentionScoreV3.md` | 1501 | doc/api | core | V3 API（MLA 等） | `3315ac15124a2bdb…` | V3 接口 |
| `docs/aclnnFusedInferAttentionScoreV4.md` | 2404 | doc/api | core | V4 API（跨 A2/A3/950） | `f9e26375c1a723f6…` | V4 接口（跨架构主力） |
| `docs/aclnnFusedInferAttentionScoreV5.md` | 2088 | doc/api | core | V5 API（仅 950；新增 qStartIdx/kvStartIdx/pseType）；参数表 L116-500+；约束 L500+ | `2e2643354f0c54c0…` | **V5 接口（主）** |

## op_host（顶层）

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_host/fused_infer_attention_score_def.cpp` | 2053 | host/def | core | `FusedInferAttentionScore` OpDef；输入输出 dtype（V1-V5 全量组合）；属性；AICore config（910b/910_93 L568-571、950 L2048-2050） | `ad64a9ad1916d8b9…` | 算子定义 |
| `op_host/fused_infer_attention_score_tiling.cpp` | 2173 | host/tiling | core | arch 分发 L100-247；分核 L291-1548；FlashDecode L1426-1439；MLA L250-263；PA L174-189；workspace L1536-1537；GenTilingKey L923-924/L1548 | `5b066c7476a5c8ae…` | tiling 主逻辑 |
| `op_host/fused_infer_attention_score_tiling.h` | 204 | host/tiling | core | tiling 数据结构/接口 | `d41320942fa7173e…` | tiling 结构体 |
| `op_host/flash_attention_infer_tiling.h` | 684 | host/tiling | core | `GetTilingKey()`；FlashAttention 推理 tiling 结构 | `f91e16bc332bffb2…` | tiling key 生成 |
| `op_host/fused_infer_attention_score_infershape.cpp` | 427 | host | core | shape/dtype 推断 | `dadc2fd28f600c00…` | infershape |
| `op_host/fused_infer_attention_score_tiling_register.cpp` | 31 | host/register | core | tiling 注册/平台查询 | `9fa6346d1b28f14d…` | tiling 注册 |
| `op_host/fused_infer_attention_score_tiling_compile_info.h` | 35 | host | core | 编译信息结构 | `053d73389fde7c47…` | compile info |
| `op_host/fused_infer_attention_score_tiling_index.h` | 79 | host | core | 分核索引常量/宏 | `cf6cf56ba65b9373…` | tiling index |
| `op_host/fallback_fused_infer_attention_score.cpp` | 332 | host/fallback | supporting | host fallback 执行 | `8d655b3e882c42bf…` | fallback |

## op_host/arch32（A2/A3 tiling + 检查）

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_host/arch32/fused_infer_attention_score_tiling_v3.cpp` | 1223 | host/tiling | core | arch32 tiling 主逻辑 + tiling key 生成 | `602bd85162ad44ff…` | arch32 tiling |
| `op_host/arch32/fused_infer_attention_score_tiling_v3.h` | 30 | host/tiling | core | arch32 tiling 结构 | `94720a5d91f9c35d…` | arch32 tiling 头 |
| `op_host/arch32/fused_infer_attention_score_tiling_info_parser.cpp` | 1173 | host | core | tiling 信息解析 | `03a93ae3b69fa24e…` | tiling info parser |
| `op_host/arch32/fused_infer_attention_score_tiling_info_parser.h` | 180 | host | core | parser 头 | `4f469221cafaaf59…` | parser 头 |
| `op_host/arch32/fused_infer_attention_score_tiling_check.cpp` | 119 | host/check | core | 通用 tiling 检查 | `e4f97f00633a182a…` | 检查入口 |
| `op_host/arch32/fused_infer_attention_score_tiling_check.h` | 324 | host/check | core | 检查接口 | `7cc757cae53db299…` | 检查头 |
| `op_host/arch32/fused_infer_attention_score_tiling_check_consistency.cpp` | 759 | host/check | core | 参数一致性检查 | `cc30ea445a4843f2…` | 一致性检查 |
| `op_host/arch32/fused_infer_attention_score_tiling_check_existence.cpp` | 702 | host/check | core | 输入存在性检查 | `a14c50403819a67d…` | 存在性检查 |
| `op_host/arch32/fused_infer_attention_score_tiling_check_feature.cpp` | 1012 | host/check | core | 特性支持检查 | `8b38487bb17fcb3e…` | 特性检查 |
| `op_host/arch32/fused_infer_attention_score_tiling_check_single_para.cpp` | 700 | host/check | core | 单参数有效性检查 | `947d88ca26e99b88…` | 单参检查 |

## op_host/arch35（950 tiling）

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_host/arch35/fused_infer_attention_score_tiling_v2.cpp` | 1156 | host/tiling | core | arch35 tiling 主逻辑；模板 tiling key L1022-1025/L1145-1148 | `65a9bf28b96e4f41…` | arch35 tiling |
| `op_host/arch35/fused_infer_attention_score_tiling_v2.h` | 36 | host/tiling | core | arch35 tiling 结构 | `3a9abc3fd53a95e6…` | arch35 tiling 头 |

## op_host/op_api（aclnn V1-V5 包装）

| 文件 | 行数 | role | centrality | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|
| `op_api/aclnn_fused_infer_attention_score.cpp/.h` | 81/50 | api | core | `18589e4d…`/`226c13be…` | V1 aclnn |
| `op_api/aclnn_fused_infer_attention_score_v2.cpp/.h` | 228/56 | api | core | `f1d64433…`/`ddb0e9d8…` | V2 aclnn |
| `op_api/aclnn_fused_infer_attention_score_v3.cpp/.h` | 254/58 | api | core | `f5cbc3ef…`/`14ebd6d6…` | V3 aclnn |
| `op_api/aclnn_fused_infer_attention_score_v4.cpp/.h` | 229/54 | api | core | `81653915…`/`59866960…` | V4 aclnn |
| `op_api/aclnn_fused_infer_attention_score_v5.cpp/.h` | 257/55 | api | core | `7764ed8c…`/`68e8b6c4…` | V5 aclnn |
| `op_api/aclnn_fused_infer_attention_score_inner.cpp` + `fused_infer_attention_score_inner.h` | 141/69 | api | core | `1efdfba8…`/`4163f072…` | 内部实现复用 |

## op_kernel（顶层入口 + 模板）

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_kernel/fused_infer_attention_score.cpp` | 667 | kernel/entry | core | arch32 入口；按 tiling key 分发 regular/decode 模板 | `735847454774c89f…` | kernel 入口（arch32） |
| `op_kernel/fused_infer_attention_score_apt.cpp` | 73 | kernel/entry | core | arch35 入口 | `d7adb0cecde4af1f…` | kernel 入口（arch35） |
| `op_kernel/fused_infer_attention_score_v3.cpp` | 3135 | kernel | core | V3 kernel 主体 | `f083f1fbe0ea689d…` | V3 kernel |
| `op_kernel/flash_attention_regular.h` | 1059 | kernel/template | core | **全量（prefill）regular 模板**（Q_S>1） | `0cd93ab0a937eba4…` | prefill 模板 |
| `op_kernel/flash_attention_regular_decode.h` | 745 | kernel/template | core | **decode 模板**（Q_S=1，含 FlashDecode） | `a879b8acc7ef4af9…` | decode 模板 |
| `op_kernel/flash_attention_interface.cpp` | 215 | kernel | core | kernel 接口层 | `2b208992b20d5d01…` | 接口层 |
| `op_kernel/fused_infer_attention_score_template_tiling_key.h` | 3116 | kernel/dispatch | core | 模板分发（`if constexpr` on layout/S/D/implMode/quant/mask/...） | `d474235a225bef56…` | 模板分发 |
| `op_kernel/fused_infer_attention_score_tilingkey.h` | 259 | kernel/dispatch | core | arch32 tilingkey 宏 | `3ad2cf44a9f00c2d…` | arch32 模板选择 |

## op_kernel/attn_infra（header-only C++ 模板库，FIA 核心架构）

`attn_infra/` 按 gemm（matmul）/ epilogue（softmax/rescale/writeback）/ layout（ND/NZ）/ arch（buffer/sync/resource）分层，对应 attention 的 QK->softmax->PV->rescale 流程。52 个 `.hpp`。

### arch/（硬件抽象：buffer/sync/resource）

| 文件 | 行数 | role | centrality | 关键符号 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `attn_infra/arch/resource.hpp` | 42 | infra-arch | core | `Resource`（UB/L1/L0 资源管理） | `7e4d4c8de5a151a9…` | 资源管理 |
| `attn_infra/arch/local_tensor_buffer.hpp` | 234 | infra-arch | core | `LocalTensorBuffer`（本地 tensor 缓冲） | `25067f5754d4070c…` | tensor 缓冲 |
| `attn_infra/arch/cross_core_sync.hpp` | 119 | infra-arch | core | `CrossCoreSync`（核间同步） | `aa4a1b40b12a37aa…` | 核间同步 |
| `attn_infra/arch/arch.hpp` | 54 | infra-arch | core | arch 调度入口 | `fc3cdec7d195977f…` | arch 入口 |

### gemm/（matmul：QK=Bmm1 / PV=Bmm2）

| 文件 | 行数 | role | centrality | 关键符号 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `attn_infra/gemm/block/block_mmad_qk.hpp` | 347 | infra-gemm | core | `BlockMmadQk`（**Bmm1: Q·K^T**，prefill） | `a1051a36153eac77…` | Bmm1 全量 |
| `attn_infra/gemm/block/block_mmad_qk_decode.hpp` | 289 | infra-gemm | core | `BlockMmadQkDecode`（Bmm1 decode） | `9feb4a3c6a5298c9…` | Bmm1 decode |
| `attn_infra/gemm/block/block_mmad_pv.hpp` | 324 | infra-gemm | core | `BlockMmadPv`（**Bmm2: P·V**，prefill） | `d0f2050d37a98c1e…` | Bmm2 全量 |
| `attn_infra/gemm/block/block_mmad_pv_decode.hpp` | 285 | infra-gemm | core | `BlockMmadPvDecode`（Bmm2 decode） | `a2d9f8e65539d2aa…` | Bmm2 decode |
| `attn_infra/gemm/block/block_mmad.hpp` | 41 | infra-gemm | core | `BlockMmad` 基类 | `08e9220a22465929…` | mmad 基类 |
| `attn_infra/gemm/tile_common/copy_gm_to_l1.hpp` | 1066 | infra-gemm | core | GM->L1 搬运（Cube 输入） | `ea8ff54fc2ecf762…` | Cube 输入搬运 |
| `attn_infra/gemm/tile_common/copy_l1_to_l0a.hpp` | 355 | infra-gemm | core | L1->L0A 搬运 | `975e8524df367d00…` | L1->L0A |
| `attn_infra/gemm/tile_common/copy_l1_to_l0b.hpp` | 486 | infra-gemm | core | L1->L0B 搬运 | `f0657a48522e9fda…` | L1->L0B |
| `attn_infra/gemm/tile_common/copy_l0c_to_gm.hpp` | 208 | infra-gemm | core | L0C->GM 搬运（Cube 输出） | `4328b7010bfd8c74…` | Cube 输出搬运 |
| `attn_infra/gemm/tile_common/tile_mmad.hpp` | 103 | infra-gemm | core | `TileMmad`（tile 级 mmad） | `8940bd3362d88146…` | tile mmad |
| `attn_infra/gemm/tile_common/tile_copy_tla.hpp` | 19 | infra-gemm | supporting | TLA tile copy | `0ca1cc4dfa7a5943…` | TLA copy |
| `attn_infra/gemm/tile_common/copy_gm_to_ub.hpp`/`copy_ub_to_gm.hpp`/`copy_l1_to_bt.hpp`/`tile_copy.hpp` | 21/21/26/62 | infra-gemm | supporting | 各搬运/copy 工具 | `84e3a803…`/`c2abaced…`/`33640ddd…`/`0a4e71fd…` | gemm 搬运工具 |
| `attn_infra/gemm/dispatch_policy.hpp`/`gemm_type.hpp`/`helper.hpp` | 75/27/254 | infra-gemm | core | dispatch policy / gemm 类型 / helper | `3b425311…`/`fc376e21…`/`648fccfd…` | gemm 策略/工具 |

### epilogue/（softmax/rescale/writeback）

| 文件 | 行数 | role | centrality | 关键符号 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `attn_infra/epilogue/block/block_epilogue_online_softmax.hpp` | 2050 | infra-epilogue | core | `BlockEpilogueOnlineSoftmax`（**FlashSoftmax 在线 max/sum/exp rescale**） | `9313fbdc40864b00…` | FlashSoftmax 核心 |
| `attn_infra/epilogue/block/block_epilogue_online_softmax_low_prec.hpp` | 867 | infra-epilogue | core | low_prec 变体（低精度） | `b7a6e5c2395a7d83…` | FlashSoftmax 低精度 |
| `attn_infra/epilogue/block/block_epilogue_rescale_o.hpp` | 1102 | infra-epilogue | core | `BlockEpilogueRescaleO`（**rescale ÷softmax_sum + 写回**） | `f62bb85afe9514c3…` | rescale 写回 |
| `attn_infra/epilogue/block/block_epilogue_rescale_o_low_prec.hpp` | 471 | infra-epilogue | core | rescale 低精度变体 | `ad91e8354207314a…` | rescale 低精度 |
| `attn_infra/epilogue/block/CombineScale.hpp` | 281 | infra-epilogue | core | `CombineScale`（scale 合并，量化场景） | `afe24821e8b87358…` | scale 合并 |
| `attn_infra/epilogue/block/block_epilogue.hpp`/`block_epilogue_init_outputs.hpp` | 33/163 | infra-epilogue | core | epilogue 基类/初始化 | `f949d074…`/`5220a4d9…` | epilogue 基类 |
| `attn_infra/epilogue/tile_common/`（13 文件） | 49-187 | infra-epilogue | core | `copy_gm_to_ub`/`copy_ub_to_gm`/`tile_broadcast_*`/`tile_cast`/`tile_copy`/`tile_elemwise_*`/`tile_swizzle` | `b272d526…` 等 | Vector 搬运/elemwise |
| `attn_infra/epilogue/dispatch_policy.hpp` | 55 | infra-epilogue | core | epilogue dispatch policy | `2730463aca36daf2…` | epilogue 策略 |

### layout/（ND/NZ 布局）

| 文件 | 行数 | role | centrality | 关键符号 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `attn_infra/layout/matrix.hpp` | 1207 | infra-layout | core | `Matrix`布局（ND/NZ 矩阵） | `a0c8b0d5390df843…` | 矩阵布局 |
| `attn_infra/layout/vector.hpp` | 132 | infra-layout | core | `Vector` 布局 | `c92ab95ba5388bdf…` | 向量布局 |
| `attn_infra/layout/layout.hpp` | 17 | infra-layout | core | layout 入口 | `9f544ae8d6c25568…` | layout 入口 |

### 顶层 + detail/（基础定义/工具）

| 文件 | 行数 | role | centrality | 关键符号 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `attn_infra/coord.hpp` | 441 | infra-common | core | `Coord`（坐标） | `5fbbb783d3f9185e…` | 坐标 |
| `attn_infra/gemm_coord.hpp` | 163 | infra-common | core | `GemmCoord` | `0b48355df72d61d3…` | gemm 坐标 |
| `attn_infra/matrix_coord.hpp` | 107 | infra-common | core | `MatrixCoord` | `24dcc6c97f8b2bb6…` | 矩阵坐标 |
| `attn_infra/base_defs.hpp` | 45 | infra-common | core | 基础定义 | `878dbbdb3203d6bc…` | 基础定义 |
| `attn_infra/detail/alignment.hpp`/`macros.hpp`/`dependent_false.hpp` | 74/15/19 | infra-common | supporting | 对齐/宏/工具 | `d707f80a…`/`3ed7388e…`/`ba52dfdf…` | 工具 |

## tests / examples / op_graph

| 文件 | 行数 | role | centrality | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|
| `tests/ut/op_host/arch35/test_fused_infer_attention_score_tiling.cpp` | 13711 | test | indirect | `6c3974be6d967852…` | arch35 tiling UT |
| `tests/ut/op_host/test_fused_infer_attention_score_infershape.cpp` | 708 | test | indirect | `5c80a7884c7f62ba…` | infershape UT |
| `tests/st/aclnnFusedInferAttentionScoreV4/executor_*.py` | 9587 | test | indirect | `9196e8f2…` | V4 ST |
| `tests/pytest/*.py` | 45-181 | test | indirect | `173dce2f…` 等 | pytest 用例 |
| `examples/test_aclnn_fused_infer_attention_score.cpp` | 254 | example | indirect | `95601bcb…` | V4 示例 |
| `examples/arch35/test_aclnn_fused_infer_attention_score_v5.cpp` | 261 | example | indirect | `47e46e14…` | V5 arch35 示例 |
| `op_graph/fused_infer_attention_score_proto.h`（如存在） | - | op_graph | supporting | - | op graph proto |

注：
- FIA 无 `算子设计介绍.md`（仅有 aclnn V1-V5 doc + README），tiling/pipeline 知识主要来自 `tiling.cpp`/`tiling_v3.cpp`/`tiling_v2.cpp` 源码与 aclnn doc；`tiling.cpp` 行号为静态读取所得，关键逻辑精确行号 `needs_verification`。
- `attn_infra/` 为 header-only 模板库，关键符号（`BlockMmadQk`/`BlockMmadPv`/`BlockEpilogueOnlineSoftmax`/`BlockEpilogueRescaleO` 等）由文件名+分层推断，精确类名/行号 `needs_verification`（以源码为准）。
- V5 仅支持 950/arch35（V5 doc L10-15），V4 及之前跨 A2/A3/950；V5 用分离的 `keyAntiquantScale`/`valueAntiquantScale` 替代 `antiquantScale`/`antiquantOffset`（后者 V5 不支持）。
