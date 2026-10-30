# MoeFinalizeRoutingV2 — 源文件清单 (S1)

来源仓库：`kernelwiki/artifacts/ops-transformer/`（CANN ops-transformer，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`，2026-07-09）。
算子目录：`moe/moe_finalize_routing_v2/`。共 ~40 个源文件。所有路径相对算子目录。

## 核心文件

| 文件 | 行数 | 角色 | 中心度 | 职责 |
|------|------|------|--------|------|
| `op_host/moe_finalize_routing_v2_def.cpp` | 339 | 定义 | core | OpDef（expanded_x/expanded_row_idx/x1/x2/bias/scales/expert_idx → y），4 个 AICore 配置（ascend910b/ascend950/ascend310p/kirin）|
| `op_host/moe_finalize_routing_v2_infershape.cpp` | 317 | infershape | supporting | 输出形状推导 |
| `op_host/moe_finalize_routing_v2_tiling.cpp` | 94 | tiling入口 | core | TilingMoeFinalizeRoutingV2 分发 |
| `op_host/moe_finalize_routing_v2_tiling.h` | 182 | tiling头 | core | MoeFinalizeRoutingV2TilingData（totalCoreNum/usedCoreNum/H/K/skip2IsNull/biasRowNum/totalRowNum/hSliceNum/kSliceNum 等）、MoeFinalizeRoutingTilingV2 基类 |
| `op_host/moe_finalize_routing_v2_tiling_membase.cpp` | 925 | tiling | core | membase 架构 tiling（CalcOpTiling、H/K 切分、核循环）|
| `op_host/moe_finalize_routing_v2_tiling_arch35.cpp` | 773 | tiling | core | arch35 regbase 架构 tiling |
| `op_kernel/moe_finalize_routing_v2.cpp` | 299 | kernel入口 | core | kernel 函数（KERNEL_TYPE_AIV_ONLY），~50 个 tiling key 分支（FP16/FLOAT/BF16 × CutK/Db/CuthK2/CuthK4/Cuth/AllBias × 有/无 bias × mix_precision）|
| `op_kernel/moe_finalize_routing_v2_common.h` | 127 | 公共 | core | 命名空间 MoeFinalizeRoutingV2、公共定义 |
| `op_kernel/moe_finalize_routing_v2_fp_db.h` | 561 | kernel模板 | core | FP double buffer 模板 |
| `op_kernel/moe_finalize_routing_v2_fp_cuth.h` | 537 | kernel模板 | core | FP cut H 模板 |
| `op_kernel/moe_finalize_routing_v2_fp_cuth_k2.h` | 454 | kernel模板 | core | FP cut H K=2 模板 |
| `op_kernel/moe_finalize_routing_v2_fp_cuth_k4.h` | 490 | kernel模板 | core | FP cut H K=4 模板 |
| `op_kernel/moe_finalize_routing_v2_fp_cutk.h` | 451 | kernel模板 | core | FP cut K 模板 |
| `op_kernel/moe_finalize_routing_v2_fp_db_all_bias.h` | 612 | kernel模板 | core | FP double buffer + all bias 模板 |
| `op_kernel/moe_finalize_routing_v2_bf16.h` | 636 | kernel模板 | core | BF16 双缓冲模板 |
| `op_kernel/moe_finalize_routing_v2_bf16_all_bias.h` | 702 | kernel模板 | core | BF16 all bias 模板 |
| `op_kernel/moe_finalize_routing_v2_bf16_cuth.h` | 621 | kernel模板 | core | BF16 cut H 模板 |
| `op_kernel/moe_finalize_routing_v2_bf16_cuth_k2.h` | 514 | kernel模板 | core | BF16 cut H K=2 模板 |
| `op_kernel/moe_finalize_routing_v2_bf16_cuth_k4.h` | 621 | kernel模板 | core | BF16 cut H K=4 模板 |
| `op_kernel/moe_finalize_routing_v2_bf16_cutk.h` | 509 | kernel模板 | core | BF16 cut K 模板 |
| `op_kernel/moe_finalize_routing_v2_fp_cuth_k2_optimized.h` | 490 | kernel模板 | core | FP cut H K=2 优化版（network）|
| `op_kernel/arch35/moe_finalize_routing_v2_base.h` | 367 | kernel模板 | core | arch35 regbase 基础模板 |
| `op_kernel/arch35/moe_finalize_routing_v2_h_full_load.h` | 259 | kernel模板 | core | arch35 H 全加载 |
| `op_kernel/arch35/moe_finalize_routing_v2_h_partial_load.h` | 224 | kernel模板 | core | arch35 H 部分加载 |
| `op_kernel/arch35/moe_finalize_routing_v2_k_h_full_load.h` | 380 | kernel模板 | core | arch35 K+H 全加载 |
| `op_kernel/arch35/moe_finalize_routing_v2_row_k_h_full_load.h` | 253 | kernel模板 | core | arch35 row+K+H 全加载 |