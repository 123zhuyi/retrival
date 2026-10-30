# block_sparse_attention_grad - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/block_sparse_attention_grad/`（反向）。`fixture: true`。
前向算子 `attention/block_sparse_attention/` 见 [[ops-transformer-block-sparse-attention]]。本反向算子共享前向的类 CUTLASS `attn_infra` 模板库（结构基本一致），但额外有反向专用件：`block_epilogue_fag_pre`/`_post`/`_simply_softmax`/`_softmaxgrad` 与 `block_mmad_bsag2`；且**不复用**前向的 `Mask2IdxAndCount` CSR 压缩，改用 mask 门控稠密扫描（见 [[kernel-block-sparse-attention-grad]] §3）。

## 核心与支撑文件

| 文件（相对算子目录） | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 373 | doc | core | 反向公式 L19-71；约束 L361-367（headdim=128、当前仅 BNSD+MHA）；产品支持 L5-14 | `1f3371af5ab28605...` | 算子总览 |
| `docs/aclnnBlockSparseAttentionGrad.md` | 763 | doc/api | core | 反向公式 L19-71；参数表 L91-320；约束；产品支持 L5-14（仅训练） | `e6039ae6b18e9de3...` | 接口语义（权威） |
| `op_host/block_sparse_attention_grad_def.cpp` | 92 | host/def | core | `BlockSparseAttentionGrad` OpDef：输入 L19-62；输出 dq/dk/dv L63-74；属性 L76-82；AICore(910b/910_93) L84-85 | `82f08000735bb871...` | 算子定义 |
| `op_host/block_sparse_attention_grad_infershape.cpp` | 94 | host | core | dq=query/dk=key/dv=value 形状直拷 L82-84 | `2bf5429bfecd36cd...` | shape 推导 |
| `op_host/block_sparse_attention_grad_tiling.h` | 191 | host/tiling | core | `BlockSparseAttentionGradTilingData` L22-77；`BSAGradTiling` L102；`SoftMaxTiling` 嵌套 L77 | `72b67854ff079a2d...` | tiling 结构体 |
| `op_host/block_sparse_attention_grad_tiling.cpp` | 603 | host/tiling | core | `GenerateTilingKey`（4 key）L497-514；`CalculateTaskSplit` L369-420；`CalculateWorkSpace` L422-447；`CalculateSoftmaxGradTiling` L278-300 | `77c2a1d037e3a82d...` | host tiling 主逻辑 |
| `op_host/op_api/aclnn_block_sparse_attention_grad.cpp` | 322 | api | core | `ValidateParams` L108-188；GetWorkspaceSize L236-307 | `178b25f5b840e80b...` | aclnn C++ API |
| `op_host/op_api/aclnn_block_sparse_attention_grad.h` | 94 | api | supporting | aclnn 声明 | `e9f1430a4cd42800...` | API 头 |
| `op_host/op_api/block_sparse_attention_grad.cpp` | 105 | api | core | l0op `BlockSparseAttentionGrad` L43-99；kv layout 默认随 q L64 | `b31cb1f4905c8b6f...` | l0op 包装 |
| `op_host/op_api/block_sparse_attention_grad.h` | 43 | api | supporting | l0op 声明 | `9bb1a6cc00cc0f14...` | API 头 |
| `op_kernel/block_sparse_attention_grad.cpp` | 60 | kernel/entry | core | `extern "C"` 入口 L24-38；4 tiling key 分派 L44-60 | `75f8e2a15551be3e...` | kernel 入口 |
| `op_kernel/block_sparse_attention_grad_kernel.h` | 741 | kernel | core | `BlockSparseAttentionGradKernel` L61；cube `operator()` L291-513；vec `operator()` L515-531；mask 门控扫描 L386-389,604-607；3 flag L39-41；workspace 分区 L341-355 | `bf48e66d0f8b102f...` | kernel 主体 |
| `op_kernel/block_sparse_attention_grad_interface.h` | 133 | kernel | core | `BlockSparseAttentionGradInfer` L27；3 cube GEMM 分组 L48/63/78；tile shape/分派策略；模板实例化 L123-125 | `24fc40ced2d3381f...` | kernel 接线/模板 |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_fag_pre.hpp` | 193 | template/epilogue | core | FAGPre：dq/dk/dv workspace 清零 L140-166 | `d67dbade6e934bd1...` | 反向 pre-stage |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_simply_softmax.hpp` | 537 | template/epilogue | core | `SimpltSoftmax`：重算 P=exp(S·scale−Lse) L335-393；dS=P·(dP−D) L402-444 | `6ef90496ebd157bc...` | 重算 P + CalDs |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_softmaxgrad.hpp` | 395 | template/epilogue | core | `SoftmaxGrad`：D=Σ(dout·out) L208-297 | `64dd0fe2db273757...` | softmax-grad 项 |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_post.hpp` | 416 | template/epilogue | core | `BlockPost`：rescale(dQ/dK*=scale)+cast+写回 L335-411 | `a5ddfb5d4b7f709f...` | 反向 post-stage |
| `op_kernel/attn_infra/epilogue/block/block_epilogue.hpp` | 32 | template/epilogue | supporting | epilogue 基类/聚合 | `a5e4374c3a2137a6...` | epilogue 基础 |
| `op_kernel/attn_infra/epilogue/dispatch_policy.hpp` | 43 | template/epilogue | supporting | `EpilogueAtlasA2FAGPre`（单一策略，无 LseMode/SM_Dtype）L38-40 | `63ade991511477da...` | epilogue 分派策略 |
| `op_kernel/attn_infra/gemm/block/block_mmad_bsag2.hpp` | 197 | template/gemm | core | `SBSAG2` 共享 BlockMmad：3 组 cube GEMM 复用；AtomicAdd 累加 L149-164 | `ae0bd2b7fd0478eb...` | 反向 cube GEMM |
| `op_kernel/attn_infra/gemm/block/block_mmad.hpp` | 38 | template/gemm | supporting | 仅 include bsag2 | `dbe61378d8c1c78e...` | GEMM 聚合 |
| `op_kernel/attn_infra/gemm/dispatch_policy.hpp` | 85 | template/gemm | supporting | `MmadAtlasA2SBSAG{1,2,3}` L72-82（仅 SBSAG2 有特化） | `d674ae48b0b13caf...` | GEMM 分派策略 |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax[_low_prec].hpp` | 1062/846 | template/epilogue | supporting | 在线 softmax（与前向共享，反向未用） | `a9c8140868d61655...` | 共享件（未用） |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o[_low_prec].hpp` | 403/390 | template/epilogue | supporting | rescale-O（与前向共享，反向未用） | `455030781827fab9...` | 共享件（未用） |
| `op_kernel/attn_infra/arch/arch.hpp` | 54 | template/arch | supporting | AtlasA2 内存容量 | `fc3cdec7d195977f...` | 架构描述 |
| `op_kernel/attn_infra/arch/cross_core_sync.hpp` | 114 | template/arch | supporting | `CrossCoreSetFlag/WaitFlag`（CANN 专有） | `71fda33b21154e9e...` | 跨核同步 |
| `op_kernel/attn_infra/arch/local_tensor_buffer.hpp` | 234 | template/arch | supporting | TBuf/LocalTensor（CANN 专有） | `25067f5754d4070c...` | 缓冲区抽象 |
| `op_kernel/attn_infra/arch/resource.hpp` | 42 | template/arch | supporting | 资源/pipe 容器 | `7e4d4c8de5a151a9...` | 资源管理 |
| `op_kernel/attn_infra/coord.hpp` | 323 | template/layout | supporting | `Coord` ND 坐标 | `5470fbd4cb5c6c12...` | 坐标类型 |
| `op_kernel/attn_infra/gemm_coord.hpp` | 163 | template/layout | supporting | `GemmCoord` | `0b48355df72d61d3...` | GEMM 坐标 |
| `op_kernel/attn_infra/matrix_coord.hpp` | 107 | template/layout | supporting | `MatrixCoord` | `24dcc6c97f8b2bb6...` | 矩阵坐标 |
| `op_kernel/attn_infra/layout/{layout,matrix,vector}.hpp` | 17/1204/132 | template/layout | supporting | RowMajor/ColMajor/nZ/zN 布局 | `9f544ae8d6c25568...` | 布局 |
| `op_kernel/attn_infra/base_defs.hpp` | 42 | template | supporting | 基础定义 | `299627988b7f24eb...` | 基础 |
| `op_kernel/attn_infra/detail/{alignment,dependent_false,macros}.hpp` | 79/19/15 | template | supporting | 对齐/工具 | `f44ea7d28b3e73ec...` | 工具 |
| `op_kernel/attn_infra/{epilogue,gemm}/tile_common/*.hpp` | - | template | supporting | copy/broadcast/cast/elemwise/mmad 原语（共 21 头） | - | tile 级原语 |
| `op_host/CMakeLists.txt` | 29 | build | indirect | host 构建 | `cc110b69d0c74bc2...` | 构建 |
| `CMakeLists.txt` | 20 | build | indirect | 算子构建 | `2ca29331c55d06bd...` | 构建 |
| `examples/test_aclnn_block_sparse_attention_grad.cpp` | 284 | test | indirect | 端到端示例 | `cdbb5a0d5ff1f9f0...` | 示例 |
| `tests/test_block_sparse_attention_grad.cpp` | 305 | test | indirect | 端到端测试 | `992cac3f20adc34a...` | 测试 |
| `tests/ut/op_host/test_block_sparse_attention_grad_{infershape,tiling}.cpp` | 132/81 | test | indirect | infershape/tiling UT | `fa0b3232b88c611d...` | 单测 |
| `tests/{ut/,ut/op_host/,}CMakeLists.txt` | 15/13/17 | build | indirect | 测试构建 | - | 构建 |

注：标 `-` 的为目录聚合，SHA-256 未逐个列出，可按路径在固定提交下复算。反向 `attn_infra` 与前向同源（多数头 SHA 一致），反向额外有 `block_epilogue_fag_pre`/`_post`/`_simply_softmax`/`_softmaxgrad`/`block_mmad_bsag2`；前向的 `block_mmad_qk`/`_pv`/`block_epilogue_online_softmax`（在线 softmax）在反向未使用（反向重算 P 而非在线 softmax）。
