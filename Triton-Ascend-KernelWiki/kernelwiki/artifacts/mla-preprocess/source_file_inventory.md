# mla_preprocess - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/mla_preprocess/`。`fixture: true`。无 arch 子目录（单架构 A2/A3）。

| 文件（相对算子目录） | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 |
|---|---:|---|---|---|---|
| `README.md` | 371 | doc | core | 功能/数据流 L10-19；公式 L20-54；参数 L56-361；约束 L363-371 | `ee3b0e00742a48fb...` |
| `docs/aclnnMlaPreprocess.md` | 1272 | doc/api | core | aclnn API 语义 | `9cc3ea3e90c1a13d...` |
| `op_host/mla_preprocess_def.cpp` | 122 | host/def | core | `MlaPreprocess` OpDef：输入 L73-96；输出 L97-100；属性 L101-113；AICore config(910b/910_93) L118-119 | `59aa04b833d7be9e...` |
| `op_host/mla_preprocess_infershape.cpp` | 111 | host | core | infershape | `4a86d754d252eb70...` |
| `op_host/mla_preprocess_tiling.cpp` | 708 | host/tiling | core | tiling 主逻辑；RmsNormQuantTiling/RopeConcatTiling/EinSumQuantTiling/SetTilingKey | `d029a7fad6431a1d...` |
| `op_host/mla_preprocess_tiling.h` | 68 | host/tiling | core | `MlaPreprocessParam`（N/headNum/cacheMode/quantMode）L27-42；`MlaPreprocessTiling` 类 L47-65 | `3d0f2c6adecab68b...` |
| `op_host/mla_preprocess_tilingdata.h` | 103 | host/tiling | core | `MlaPpMatmulTilingData`(m/k/n/m0/k0/n0/loop/swizzle) L25-43；`MlaTilingData`(numCore/mm1/mm2/mm3/rms1/rms2/esq*) L46-98 | `cd63a365c0ebb278...` |
| `op_kernel/mla_preprocess.cpp` | 1206 | kernel/entry | core | dtype dispatch(FP16/BF16) L17-22；TILING_KEY 分发(cacheMode×format) L46+；ProcessCube/ProcessVector L52-53；workspace s1/s2/s3 L41-43 | `742fed9146ac7869...` |
| `op_kernel/mla_preprocess_fp16.h` | 2656 | kernel | core | FP16 模板 `MLAOperation`（matmul+RmsNorm+RoPE+Cache） | `76860e49f3e9a0f0...` |
| `op_kernel/mla_preprocess_bf16.h` | 3093 | kernel | core | BF16 模板 | `9ff99ca4483551f6...` |
| `op_kernel/mla_preprocess_no_quant.h` | 2365 | kernel | core | no_quant 模板（浮点输出） | `05704a08e0891053...` |
| `op_kernel/mla_kernel_utils.h` | 397 | kernel/common | supporting | kernel 工具 | `cdb4ee7c156907cd...` |
| `op_kernel/mla_mma.h` | 89 | kernel/common | supporting | MMA matmul 封装 | `9024e68902ff832a...` |
| `op_kernel/mla_simd.h` | 425 | kernel/common | supporting | SIMD/Vector 计算 | `c393aca1041b0a32...` |
| `op_kernel/mla_iterator.h` | 125 | kernel/common | supporting | 迭代器基类 | `c1335c04777d2e78...` |
| `op_kernel/mla_mem.h` | 84 | kernel/common | supporting | 内存管理 | `27212387d6a6ba09...` |
| `op_kernel/mla_common.h` / `mla_common_func.h` | 24/116 | kernel/common | supporting | 公共常量/函数 | `00b12b67...` / `3199b6c2...` |
| `op_kernel/mla_hardware.h` / `mla_layout.h` / `mla_utils.h` | 36/15/72 | kernel/common | supporting | 硬件/layout/工具 | `0b39edaf...` / `a7942aa5...` / `f1058f2c...` |
| `op_kernel/{gm_to_l1,gm_to_ub,l0c_to_gm,l0c_to_l1,l0c_to_ub,l1_to_l0,l1_to_ub,l1_to_bt,l1_to_fb}_iterator.h` | 24-306 | kernel/common | supporting | CANN buffer 搬运迭代器（GM/L1/L0C/UB/BT/FB 间） | 见前述 SHA-256 |
| `tests/ut/op_host/test_mla_preprocess_{infershape,tiling}.cpp` | 107/115 | test | indirect | UT | `db1b35d4...` / `f7c89915...` |
| `tests/ut/op_kernel/test_mla_preprocess.cpp` | 258 | test | indirect | kernel UT | `7639a217...` |

注：`*_iterator.h`（gm_to_l1/l0c_to_gm/l1_to_l0 等）是 CANN 专用 buffer 搬运迭代器，属 CANN 专有机制，不迁移。完整 SHA-256 见本仓库 `kernelwiki/artifacts/ops-transformer/attention/mla_preprocess/` 实测。
