# lightning_indexer - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/lightning_indexer/`。`fixture: true`。
LightningIndexer 是稀疏注意力（SparseFlashAttention）的**索引选择器**：根据 Q_index 和 K_index 计算重要性分数，经 ReLU 和 weights 缩放后 topK 选出每个 token 对应的最重要 KV 位置（sparse_indices），并可选输出 sparse_values。

| 文件（相对算子目录） | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 196 | doc | core | 公式 L20-24；参数 L39-71；约束 L80-86；产品支持 L5-12 | `f19f9dbca76399be...` | 算子总览 |
| `docs/aclnnLightningIndexer.md` | 740 | doc/api | core | 公式；参数表；约束 | `dd1e5ec1b59c1e42...` | 接口语义 |
| `op_host/lightning_indexer_def.cpp` | 79 | host/def | core | `LightningIndexer` OpDef L19-77；输入 query/key/weights L23-37；输出 sparse_indices/values L53-60；属性 L61-67；AICore(910b/910_93) L75-76 | `adeea6351c871607...` | 算子定义 |
| `op_host/lightning_indexer_infershape.cpp` | 112 | host | core | sparse_indices shape `[B,S1,N2,sparseCount]` / `[T1,N2,sparseCount]`；sparse_values 同 shape | `9e5107b0f8777fdd...` | shape 推导 |
| `op_host/lightning_indexer_tiling.h` | 239 | host/tiling | core | `LITilingData` L79-93；`LITilingInfo` L121-154；`LIInfoParser` L157-226；`LightningIndexerTiling` L229-237 | `9ba89b537c8def48...` | tiling 结构体 |
| `op_host/lightning_indexer_tiling.cpp` | 819 | host/tiling | core | tiling 主逻辑；核心分区；`HEAD_DIM_LIMIT=128` L73；`SPARSE_LIMIT=2048` L74 | `7107286ee1eeb32c...` | host tiling |
| `op_host/op_api/aclnn_lightning_indexer.cpp` | 137 | api | core | aclnn API 包装 | `33155ed8b4ba0110...` | API |
| `op_kernel/lightning_indexer.cpp` | 54 | kernel/entry | core | `extern "C"` 入口 L33-54；`KERNEL_TYPE_MIX_AIC_1_2` L45；tiling key 模板分派 L46-51 | `0f2678ed9cf5ebde...` | kernel 入口 |
| `op_kernel/lightning_indexer_kernel.h` | 677 | kernel | core | `LIPreload` 调度器 L53-153；`SplitCore` L279-357；`ProcessMain`/`ProcessBaseBlock`/`ProcessDecode` L140-143 | `8e757fd849a6bee3...` | kernel 主体 |
| `op_kernel/lightning_indexer_service_cube.h` | 420 | kernel/cube | core | `LIMatmul` L28-80；`ComputeMm1` 主逻辑；`Fixp`/`ComuteL0c`/`KeyNd2NzForPA` L68-77 | `12b8bc0b93318b38...` | Cube GEMM 服务 |
| `op_kernel/lightning_indexer_service_vector.h` | 700 | kernel/vector | core | `LIVector` L50-112；`ProcessVec` L65；`ProcessLD`（topK + decode）L66；`CopyIn`/`DoScale`/`TopK`/`SortAndSelect` 等 | `938d11abc7002fbc...` | Vector 服务 |
| `op_kernel/lightning_indexer_vector.h` | 423 | kernel/vector | core | `CopyIn` L43-66；`DoScale` L82-117；`SortAndSelect`/`TopK` 等 | `83ca9b377bad6e22...` | Vector 原语 |
| `op_kernel/lightning_indexer_template_tiling_key.h` | 87 | kernel/dispatch | core | 模板参数：DT_Q/K/OUT、PAGE_ATTENTION、LAYOUT_T/K、DT_W_FLAG；24 合法组合 L45-86 | `a087348920a255ae...` | 模板分派 key |
| `op_kernel/lightning_indexer_common.h` | 148 | kernel/common | core | `LIType` L27-38；`ConstInfo` L64-112；`SplitCoreInfo` L114-122；`RunInfo` L40-62 | `bb68764d3d11d72d...` | 公共类型常量 |
| `examples/test_aclnn_lightning_indexer.cpp` | 323 | test | indirect | 端到端示例 | `cb33f2500cca02ff...` | 示例 |
| `tests/ut/op_api/test_aclnn_lightning_indexer.cpp` | 147 | test | indirect | API UT | `cb552cbf6afa6e97...` | 单测 |
| `tests/ut/op_host/arch32/test_lightning_indexer_tiling.cpp` | 263 | test | indirect | tiling UT | `3bf25dfbc5b2c70a...` | 单测 |
| `tests/ut/op_host/test_lightning_indexer_infershape.cpp` | 180 | test | indirect | infershape UT | `ecd523524a961472...` | 单测 |