# block_sparse_attention - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/block_sparse_attention/`（前向）。`fixture: true`。
反向算子 `attention/block_sparse_attention_grad/` 为独立算子，本清单不覆盖（可单独入库）。
注：本算子内部代号 **RFA = Rain Fusion Attention**（旧名），`aclnn_rain_fusion_attention.h` 为公开 `aclnnBlockSparseAttention*` API 的别名头；早期另有 `FASparse`（Flash-Attention Sparse）别名（`kernel_common.hpp` L175-176，标注逐步废弃）。kernel 基于类 CUTLASS 的 `attn_infra` 模板库实现。

## 核心与支撑文件

| 文件（相对算子目录） | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 178 | doc | core | 功能/约束 L156-166；块稀疏 pattern L123-146；CATLASS 模板库 L5；A2 硬件 L12-13 | `335c73fac8db1247...` | 算子总览 |
| `docs/aclnnBlockSparseAttention.md` | 938 | doc/api | core | aclnn API 语义；公式 L18-22；参数表 L77-401；约束 L487-509；产品支持 L5-12 | `9f556f91eb343cdd...` | 接口语义（权威） |
| `op_host/block_sparse_attention_def.cpp` | 83 | host/def | core | `BlockSparseAttention` OpDef：输入 L19-54；输出 L55-62；属性 L64-73；AICore config(ascend910b/910_93) L75-76 | `577fa85542a97093...` | 算子定义 |
| `op_host/block_sparse_attention_infershape.cpp` | 224 | host | core | attentionOut=query L105；softmaxLse TND[T,N,1]/BNSD[B,N,S,1] L183-201；layout/头数校验 L112-181 | `561252f1abb376f8...` | shape 推导 |
| `op_host/block_sparse_attention_tiling.h` | 218 | host/tiling | core | `BlockSparseAttentionTilingData` L22-72；`BSATiling` L120；枚举 RFAKvCacheLayout/RFAQInputLayout/BsaInnerCalcPrec L101-117 | `88949353e01be85d...` | tiling 结构体 |
| `op_host/block_sparse_attention_tiling.cpp` | 829 | host/tiling | core | `GetBsaTiling` L750；`GenerateTilingKey` L683-748；`CalculateTaskSplit` L557-603；`CalculateWorkSpace` L605-630；`CheckSparsePattern` L390-437；`GetQBlocks` L103-110 | `503b9b85b05223d5...` | host tiling 主逻辑 |
| `op_host/op_api/aclnn_block_sparse_attention.cpp` | 304 | api | core | `GetWorkspaceSize` 校验 L65-205；`MakeContiguous` L148-187；aclnnGetWorkspaceSize L214-289 | `d6e6c59c552e3fc7...` | aclnn C++ API |
| `op_host/op_api/aclnn_block_sparse_attention.h` | 56 | api | supporting | aclnn 声明 | `14de0a9a7a9ee64b...` | API 头 |
| `op_host/op_api/block_sparse_attention.cpp` | 118 | api | core | l0op `BlockSparseAttention` L43-115；`ConvertIntArrayToTensor` L28-41；`INFER_SHAPE` L89-101；kv layout 默认随 q L65 | `22b48908a56feaed...` | l0op 包装 |
| `op_host/op_api/block_sparse_attention.h` | 44 | api | supporting | l0op 声明 | `5dab8ff28bc916b5...` | API 头 |
| `op_kernel/block_sparse_attention.cpp` | 97 | kernel/entry | core | `extern "C"` 入口 L16-19；`KERNEL_TYPE_MIX_AIC_1_2` L23；tiling key 预处理分派 L29-94 | `8088a81373fe3eb6...` | kernel 入口 |
| `op_kernel/block_sparse_attention_kernel.h` | 797 | kernel | core | `BlockSparseAttentionKernel` L40-48；**`Mask2IdxAndCount` L93-248**；主循环 L447-731；cross-core flag L782-784；workspace 分区 L305-321 | `509d1101498ad7b4...` | kernel 主体 |
| `op_kernel/block_sparse_attention_tilingkey.h` | 68 | kernel/dispatch | core | `RFA_BASE_TILING` L28；12 个 tiling key 常量 L31-65 | `28366ca89f50b2ff...` | 模板分派 key |
| `op_kernel/block_sparse_attention_interface.cpp` | 113 | kernel | core | `BlockSparseAttentionInfer` L24-44；tile shape/分派策略 L68-90；模板实例化 L99-110；`ArchTag=AtlasA2` L45 | `851ffe7c37587cc6...` | kernel 接线/模板 |
| `op_kernel/kernel_common.hpp` | 179 | kernel/common | core | 同步事件 id L34-37；块常量(BASIC_BLOCK_SIZE=128/PRE_LAUNCH=2/WORKSPACE_BLOCK_SIZE_DB=131072) L39-84；枚举 L98-109；`BlockSparseAttentionKernelParams` L137-173；`FASparseKernelParams` 别名 L175-176 | `8310a58a2145e39d...` | 公共常量/参数结构 |
| `op_kernel/attn_infra/arch/arch.hpp` | 54 | template/arch | supporting | AtlasA2 内存容量(UB192KB/L1 512KB/L0A 64KB/L0B 64KB/L0C 128KB)；TPosition 映射 | `fc3cdec7d195977f...` | 架构描述 |
| `op_kernel/attn_infra/arch/cross_core_sync.hpp` | 114 | template/arch | supporting | `CrossCoreSetFlag`/`CrossCoreWaitFlag`；barrier id L22-24（CANN 专有） | `71fda33b21154e9e...` | 跨核同步原语 |
| `op_kernel/attn_infra/arch/local_tensor_buffer.hpp` | 234 | template/arch | supporting | TBuf/LocalTensor（UB/L1/L0A/L0B/L0C）（CANN 专有） | `25067f5754d4070c...` | 缓冲区抽象 |
| `op_kernel/attn_infra/arch/resource.hpp` | 42 | template/arch | supporting | 资源/pipe 容器 | `7e4d4c8de5a151a9...` | 资源管理 |
| `op_kernel/attn_infra/gemm/dispatch_policy.hpp` | 73 | template/gemm | supporting | `MmadAtlasA2SFAIQK`/`MmadAtlasA2SFAIPV` 分派策略 | `6f120cea36158d2c...` | GEMM 分派策略 |
| `op_kernel/attn_infra/gemm/block/block_mmad_qk.hpp` | 323 | template/gemm | core | `BlockMmadQK::operator()` L152-227：沿 selectIdx 稀疏遍历 KV 块，仅加载选中块 | `918e6844391bee7f...` | QK 块 mmad（稀疏 gather） |
| `op_kernel/attn_infra/gemm/block/block_mmad_pv.hpp` | 289 | template/gemm | core | `BlockMmadPV::operator()` L121-186：同稀疏遍历加载 V | `91c1de54e87d4071...` | PV 块 mmad（稀疏 gather） |
| `op_kernel/attn_infra/gemm/block/block_mmad.hpp` | 39 | template/gemm | supporting | 块 mmad 基类 | `ba17fe6b4fe6142b...` | GEMM 基础 |
| `op_kernel/attn_infra/gemm/helper.hpp` | 254 | template/gemm | supporting | `L1AlignHelper`；`ElementAccumulatorSelector`；layout 映射 | `648fccfd0e48bc9a...` | GEMM 辅助 |
| `op_kernel/attn_infra/gemm/gemm_type.hpp` | 27 | template/gemm | supporting | GEMM 类型定义 | `fc376e219f8dac64...` | 类型 |
| `op_kernel/attn_infra/epilogue/dispatch_policy.hpp` | 38 | template/epilogue | supporting | `EpilogueAtlasA2OnlineSoftmax`/`EpilogueAtlasA2RescaleO`（lseMode, SoftmaxDtype） | `641b83d9ad6b572a...` | epilogue 分派策略 |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp` | 1070 | template/epilogue | core | 在线 softmax（FP32 路径）：running max/sum L280-392；rescale；`BlockReduceMax/Sum<float>` | `1caea0f2c30a0ff9...` | 在线 softmax epilogue（fp32） |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax_low_prec.hpp` | 859 | template/epilogue | core | 在线 softmax（FP16 路径，innerPrecise=1） | `c95797b5f32f2b03...` | 在线 softmax epilogue（fp16） |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o.hpp` | 460 | template/epilogue | core | O rescale/写回（FP32）；LSE 输出 L295 | `76d1f242ce051454...` | rescale-O epilogue（fp32） |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o_low_prec.hpp` | 462 | template/epilogue | core | O rescale/写回（FP16） | `27989aaa5a5f0083...` | rescale-O epilogue（fp16） |
| `op_kernel/attn_infra/epilogue/block/block_epilogue.hpp` | 31 | template/epilogue | supporting | epilogue 基类 | `277cc28abac2edd5...` | epilogue 基础 |
| `op_kernel/attn_infra/coord.hpp` | 441 | template/layout | supporting | `Coord` ND 坐标算术 | `5fbbb783d3f9185e...` | 坐标类型 |
| `op_kernel/attn_infra/gemm_coord.hpp` | 163 | template/layout | supporting | `GemmCoord`（M/N/K） | `0b48355df72d61d3...` | GEMM 坐标 |
| `op_kernel/attn_infra/matrix_coord.hpp` | 107 | template/layout | supporting | `MatrixCoord`（行/列） | `24dcc6c97f8b2bb6...` | 矩阵坐标 |
| `op_kernel/attn_infra/layout/matrix.hpp` | 1204 | template/layout | supporting | RowMajor/ColMajor/nZ/zN/Padding 布局 | `a14b0e542d6e2ad6...` | 矩阵布局 |
| `op_kernel/attn_infra/layout/vector.hpp` | 132 | template/layout | supporting | 1D 向量布局 | `67f87c7915b1545d...` | 向量布局 |
| `op_kernel/attn_infra/base_defs.hpp` | 42 | template | supporting | 基础定义 | `299627988b7f24eb...` | 基础 |
| `op_kernel/attn_infra/detail/alignment.hpp` | 79 | template | supporting | 对齐辅助 | `f44ea7d28b3e73ec...` | 对齐 |
| `op_kernel/attn_infra/detail/dependent_false.hpp` | 19 | template | supporting | `dependent_false`（static_assert） | `ba52dfdf5f7c68ed...` | 工具 |
| `op_kernel/attn_infra/detail/macros.hpp` | 15 | template | supporting | 宏 | `ddaf580b518e4def...` | 工具 |
| `op_kernel/attn_infra/epilogue/tile_common/*.hpp` | — | template/epilogue | supporting | copy GM↔UB / broadcast / cast / elemwise mul·add / swizzle（共 11 个头） | — | tile 级原语 |
| `op_kernel/attn_infra/gemm/tile_common/*.hpp` | — | template/gemm | supporting | copy GM→L1/L1→L0A·L0B/L0C→GM / tile_mmad / tile_copy（共 10 个头） | — | tile 级搬运/mmad 原语 |
| `op_kernel/attn_infra/layout/layout.hpp` | 17 | template/layout | supporting | layout 入口 | `9f544ae8d6c25568...` | layout 入口 |
| `op_host/CMakeLists.txt` | 29 | build | indirect | host 构建 | `f9df3bcd8120019a...` | 构建 |
| `CMakeLists.txt` | 20 | build | indirect | 算子构建 | `2ca29331c55d06bd...` | 构建 |
| `op_host/OWNERS` | 59 | gov | indirect | 维护者 | `5acf947314ce52da...` | 治理 |
| `examples/test_aclnn_rain_fusion_attention.cpp` | 437 | test | indirect | rain_fusion 别名端到端示例 | `ef2ab95ee67eb36e...` | 示例 |
| `tests/test_rain_fusion_attention.cpp` | 307 | test | indirect | 端到端测试（含 rain_fusion 头别名） | `8395bb50deec6b76...` | 测试 |
| `tests/README.md` | 86 | test/doc | indirect | 测试说明 | `ac9de3cb43b337ee...` | 测试文档 |

注：标 `—` 的为目录聚合（tile_common 下多个小头），SHA-256 未逐个列出，可按上表路径在固定提交下复算。`attn_infra/` 为类 CUTLASS 模板库，与反向算子 `block_sparse_attention_grad` 共享同一套 `attn_infra`（结构基本一致，grad 额有 `block_epilogue_softmaxgrad`/`block_epilogue_fag_pre`/`block_epilogue_post`/`block_mmad_bsag2` 等反向专用件）。
