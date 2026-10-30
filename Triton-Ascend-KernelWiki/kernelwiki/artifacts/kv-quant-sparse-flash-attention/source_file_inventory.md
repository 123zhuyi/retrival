# kv_quant_sparse_flash_attention - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/kv_quant_sparse_flash_attention/`。`fixture: true`。源码 ~10k 行。

centrality：core = 直接实现该算子；supporting = 公共/框架依赖；indirect = 测试/示例/文档。SHA-256 为前 16 位。

KV Quant Sparse Flash Attention 是 `sparse_flash_attention` 的 KV 量化版本（Per-Token-Head-Tile-128 量化），仅推理场景（A2/A3 推理），三架构（arch32/A2 + arch35/950）。

## docs

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 261 | doc | core | 功能说明（SFA + KV 量化）L12-21；公式 L14-20；函数原型 L23-27；参数表 L29-84；约束 L90-98 | `ef0c7ddcb217b0a6…` | 算子总览 |

## op_host

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_host/kv_quant_sparse_flash_attention_def.cpp` | 155 | host/def | core | `KvQuantSparseFlashAttention` OpDef：query FP16/BF16 L23-27；key/value INT8 L28-31；AICore config（910b L92、910_93 L93、950 L151）；950 架构额外支持 FP8_E4M3FN/HIFLOAT8 L103-104 | `f9b3a97047197616…` | 算子定义 |
| `op_host/kv_quant_sparse_flash_attention_infershape.cpp` | 81 | host | core | `InferShapeKvQuantSparseFlashAttention` + `InferDataType` | `1c448b2cdf11e516…` | shape/dtype 推导 |
| `op_host/kv_quant_sparse_flash_attention_tiling.cpp` | 1855 | host/tiling | core | tiling 主逻辑（arch 分发/分核/MLA/PA/workspace/GenTilingKey） | `d5ec7a3dea4dd0d4…` | tiling 主逻辑 |
| `op_host/kv_quant_sparse_flash_attention_tiling.h` | 622 | host/tiling | core | `KvQuantSparseFlashAttentionTilingData` 结构体 | `11396aaa24218be4…` | tiling 结构体 |

## op_kernel

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_kernel/kv_quant_sparse_flash_attention.cpp` | 72 | kernel/entry | core | `kv_quant_sparse_flash_attention<FLASH_DECODE,LAYOUT_T,KV_LAYOUT_T,TEMPLATE_MODE>` 模板 kernel L37-73；arch35 分支（__CCE_AICORE__==310）L49-60；`QSFA_OP_IMPL` 宏 L26-35 | `1cd8024ff0a89b11…` | kernel 入口（4 模板参数） |
| `op_kernel/kv_quant_sparse_flash_attention_template_tiling_key.h` | 57 | kernel/dispatch | core | 模板参数声明（FLASH_DECODE/LAYOUT_T/KV_LAYOUT_T/TEMPLATE_MODE）；支持组合（BSND+PA、TND+PA）×V_TEMPLATE | `63f161d643a30e6e…` | 模板参数 + dispatch |
| `op_kernel/kv_quant_sparse_flash_attention_common.h` | 220 | kernel/common | core | 公共定义/类型/常量 | `51f7b88cdc182662…` | 公共定义 |
| `op_kernel/kv_quant_sparse_flash_attention_kernel_mla.h` | 944 | kernel/template | core | `KvQuantSparseFlashAttentionMla` 模板类（MLA-absorb 模式 kernel） | `0f52f705d4597095…` | arch32 kernel 核心 |
| `op_kernel/kv_quant_sparse_flash_attention_service_cube_mla.h` | 941 | kernel/service | core | Cube 服务（Bmm1 Q·K^T + Bmm2 P·V，MLA-absorb 模式） | `ab3608f464d73c57…` | arch32 Cube 服务 |
| `op_kernel/kv_quant_sparse_flash_attention_service_vector_mla.h` | 1299 | kernel/service | core | Vector 服务（softmax/rescale/sparse KV gather/quant） | `8bee3d0a4cb35294…` | arch32 Vector 服务 |
| `op_kernel/arch35/kv_quant_sparse_flash_attention_kernel_mla.h` | 942 | kernel/template | core | arch35 MLA-absorb kernel（FP8_E4M3FN/HIFLOAT8 支持） | `866dc0a14e6a0a8e…` | arch35 kernel 核心 |
| `op_kernel/arch35/kv_quant_sparse_flash_attention_service_cube_mla.h` | 935 | kernel/service | core | arch35 Cube 服务 | `82883fd729c50980…` | arch35 Cube |
| `op_kernel/arch35/kv_quant_sparse_flash_attention_service_vector_mla.h` | 1299 | kernel/service | core | arch35 Vector 服务 | `a51a8f81f41493a3…` | arch35 Vector |

## tests

| 文件 | 行数 | role | centrality | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|
| `tests/ut/op_host/arch32/test_kv_quant_sparse_flash_attention_tiling.cpp` | 830 | test | indirect | `93a067190c8e5825…` | arch32 tiling UT |
| `tests/ut/op_host/arch35/test_kv_quant_sparse_flash_attention_tiling.cpp` | 167 | test | indirect | `5005dfcce2f99f3e…` | arch35 tiling UT |
| `tests/ut/op_host/test_kv_quant_sparse_flash_attention_infershape.cpp` | 152 | test | indirect | `e90ec01ae76831c0…` | infershape UT |

注：
- KV Quant SFA 的 kernel 入口有 4 个模板参数：`FLASH_DECODE(bool)` / `LAYOUT_T`(BSND/TND) / `KV_LAYOUT_T`(BSND/TND/PA_BSND) / `TEMPLATE_MODE`(C/V)。支持 5 种参数组合。
- kernel 实现为 MLA-absorb 模式（attention_mode=2），Q/K D 维包含 nope+rope 拼接（Q: 512+64=576，K: 512+64*2+4*4=656）。
- arch32（A2）支持 FP16/BF16 query + INT8 KV；arch35（950）支持 BF16 query + FP8_E4M3FN/HIFLOAT8 KV。
- 仅推理场景，无设计文档（仅有 README）。