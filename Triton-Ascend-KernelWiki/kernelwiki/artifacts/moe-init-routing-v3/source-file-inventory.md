# MoeInitRoutingV3 - 源文件清单 (S1)

来源仓库：`kernelwiki/artifacts/ops-transformer/`（固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`，2026-07-09）。
算子目录：`moe/moe_init_routing_v3/`。共 ~50 个源文件。

## 核心文件

| 文件 | 行数 | 角色 | 中心度 | 职责 |
|------|------|------|--------|------|
| `README.md` | 260 | doc | core | 算子总览、6 步公式、I/O、约束、V2->V3 变更 |
| `docs/aclnnMoeInitRoutingV3.md` | 841 | doc | core | aclnn 接口文档 |
| `op_host/moe_init_routing_v3_def.cpp` | 196 | 定义 | core | OpDef（x/expert_idx/scale/offset -> expanded_x/expanded_row_idx/expert_tokens_count_or_cumsum/expanded_scale），2 套 AICore 配置（ascend910b/910_93 + ascend950 regbase）|
| `op_host/moe_init_routing_v3_infershape.cpp` | 863 | infershape | supporting | 输出形状推导 |
| `op_host/moe_init_routing_v3_tiling.cpp` | 1300 | tiling入口 | core | TilingMoeInitRoutingV3 分发（membase）|
| `op_host/moe_init_routing_v3_tiling.h` | 149 | tiling头 | core | TilingData 结构体 |
| `op_host/moe_init_routing_v3_tiling_arch35.cpp` | 1296 | tiling | core | arch35 regbase tiling |
| `op_kernel/moe_init_routing_v3.cpp` | 438 | kernel入口 | core | kernel 函数（KERNEL_TYPE_MIX_AIV_1_0），~30 tiling key 分支 |
| `op_kernel/moe_v3_common.h` | 109 | 公共 | core | 命名空间 MoeInitRoutingV3、公共定义 |
| `op_kernel/moe_v3_full_load.h` | 280 | kernel模板 | core | FullLoad 高性能模板 |
| `op_kernel/moe_v3_full_load_base.h` | 514 | kernel模板 | core | FullLoad 基类 |
| `op_kernel/moe_v3_full_load_dynamic_quant.h` | 301 | kernel模板 | core | FullLoad 动态量化 |
| `op_kernel/moe_v3_full_load_static_quant.h` | 231 | kernel模板 | core | FullLoad 静态量化 |
| `op_kernel/moe_v3_full_load_unquantized.h` | 223 | kernel模板 | core | FullLoad 非量化 |
| `op_kernel/moe_v3_sort_one_core.h` | 166 | kernel模板 | core | 单核排序 |
| `op_kernel/moe_v3_sort_multi_core.h` | 392 | kernel模板 | core | 多核排序 |
| `op_kernel/moe_v3_mrgsort.h` | 206 | kernel辅助 | core | 归并排序 |
| `op_kernel/moe_v3_mrgsort_out.h` | 231 | kernel辅助 | core | 归并排序输出 |
| `op_kernel/moe_v3_row_idx_gather.h` | 203 | kernel模板 | core | row_idx gather |
| `op_kernel/moe_v3_gather_out.h` | 321 | kernel模板 | core | gather 输出 |
| `op_kernel/moe_v3_gather_dynamic_quant.h` | 609 | kernel模板 | core | 动态量化 gather |
| `op_kernel/moe_v3_gather_static_quant.h` | 334 | kernel模板 | core | 静态量化 gather |
| `op_kernel/moe_v3_expert_tokens_count.h` | 376 | kernel模板 | core | 专家 token 计数（直方图）|
| `op_kernel/arch35/moe_v3_*` | - | kernel模板 | core | arch35 regbase 各模块（gather_dynamic_quant 531、gather_mxfp8_quant 586、gather_hif8_* 等）|