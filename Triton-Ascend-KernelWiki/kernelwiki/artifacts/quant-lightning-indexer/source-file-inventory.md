# QuantLightningIndexer — 源文件清单 (S1)

来源仓库：`kernelwiki/artifacts/ops-transformer/`（CANN ops-transformer，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`，2026-07-09）。
算子目录：`attention/quant_lightning_indexer/`。共 ~20 个源文件。所有路径相对算子目录。

## op_host（Host 侧 tiling/定义/inferShape）

| 文件 | 行数 | 角色 | 中心度 | 关键符号 / 职责 |
|------|------|------|--------|----------------|
| `op_host/quant_lightning_indexer_def.cpp` | 138 | 定义 | core | `QuantLightningIndexer` OpDef：I/O dtype/layout（int8/fp8）、属性（sparse_count/sparse_mode/layout_query/layout_key）、AICore 配置 ascend910b/910_93(int8) + ascend950(fp8) L20-136 |
| `op_host/quant_lightning_indexer_infershape.cpp` | 95 | infershape | supporting | 输出 `sparse_indices` 形状推导 |
| `op_host/quant_lightning_indexer_tiling.cpp` | 872 | tiling | core | `QuantLightningIndexerTiling::DoTiling`：`LI_LAYOUT` 宏、`DoTilingByArch` 分发、`CalcS1S2`、`CalcCoreNum`、`GetSuperGroupNum`、PageAttention 场景处理 |
| `op_host/quant_lightning_indexer_tiling.h` | 240 | tiling头 | core | `QLITilingData`（bSize/n2Size/gSize/s1Size/s2Size/sparseCount/usedCoreNum/blockSize/maxBlockNumPerBatch/sparseMode L84-95）、`QLITilingInfo`、`QLIInfoParser`、`QuantLightningIndexerTiling` 类 |

## op_kernel（Kernel 侧）

| 文件 | 行数 | 角色 | 中心度 | 关键符号 / 职责 |
|------|------|------|--------|----------------|
| `op_kernel/quant_lightning_indexer.cpp` | 62 | kernel入口 | core | kernel 函数 `quant_lightning_indexer<DT_Q,DT_K,DT_OUT,PAGE_ATTENTION,Q_LAYOUT_T,K_LAYOUT_T>` L39-62；`KERNEL_TYPE_MIX_AIC_1_2` L48；invoke `QLIPreload` L52-61 |
| `op_kernel/quant_lightning_indexer_kernel.h` | 713 | kernel模板 | core | `QLIPreload` 类：`Init`、`Process`（`QKCompute` + `IndexCompute` + `WriteBack`）、`QuantDequant`、`Matmul`、`TopK` |
| `op_kernel/quant_lightning_indexer_common.h` | 145 | 公共 | core | `QLIType` 模板结构体（`quantTypeQ`/`quantTypeK`/`outputType`/`pageAttention`/`qLayout`/`kLayout`）、`LI_LAYOUT` 宏 |
| `op_kernel/quant_lightning_indexer_template_tiling_key.h` | 98 | dispatch | core | `ASCENDC_TPL_ARGS_DECL`：6 模板参数（DT_Q/DT_K/DT_OUT/PAGE_ATTENTION/Q_LAYOUT_T/K_LAYOUT_T）；arch32 2 组合（int8 × PA_BSND/BSND/TND）；arch35 6 组合（fp8_e4m3fn/hifloat8 × PA_BSND/BSND/TND） |
| `op_kernel/quant_lightning_indexer_vector.h` | 192 | kernel辅助 | core | `QuantDequant` 反量化向量算子 |
| `op_kernel/quant_lightning_indexer_service_cube.h` | 579/612 | kernel辅助 | core | Cube 服务（matmul 封装） |
| `op_kernel/quant_lightning_indexer_service_vector.h` | 664/677 | kernel辅助 | core | Vector 服务（index compute、topk） |
| `op_kernel/arch35/quant_lightning_indexer_kernel.h` | 713 | kernel模板 | core | arch35 专用 kernel 模板（fp8 路径） |
| `op_kernel/arch35/quant_lightning_indexer_service_cube.h` | 579 | kernel辅助 | core | arch35 Cube 服务 |
| `op_kernel/arch35/quant_lightning_indexer_service_vector.h` | 677 | kernel辅助 | core | arch35 Vector 服务 |

## tests

| 文件 | 行数 | 角色 | 中心度 | 职责 |
|------|------|------|--------|
| `tests/ut/op_host/arch32/test_quant_lightning_indexer_tiling.cpp` | 239 | test | indirect | arch32 tiling 单测 |
| `tests/ut/op_host/arch35/test_quant_lightning_indexer_tiling.cpp` | 134 | test | indirect | arch35 tiling 单测 |
| `tests/ut/op_host/test_quant_lightning_indexer_infershape.cpp` | 132 | test | indirect | inferShape 单测 |