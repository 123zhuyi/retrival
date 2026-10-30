# MoeGatingTopKSoftmaxV2 - 源文件清单 (S1)

来源仓库：`kernelwiki/artifacts/ops-transformer/`（固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`，2026-07-09）。
算子目录：`moe/moe_gating_top_k_softmax_v2/`。共 ~28 个源文件。

## 核心文件

| 文件 | 行数 | 角色 | 中心度 | 职责 |
|------|------|------|--------|------|
| `README.md` | 122 | doc | core | 算子总览、renorm 0/1 公式、I/O、约束 |
| `docs/aclnnMoeGatingTopKSoftmaxV2.md` | 430 | doc | core | aclnn 接口文档 |
| `op_host/moe_gating_top_k_softmax_v2_def.cpp` | 69 | 定义 | core | OpDef（x/finished -> y/expert_idx/softmax_result），属性 k/renorm/output_softmax_result_flag，2 套 AICore 配置 |
| `op_host/moe_gating_top_k_softmax_v2_infershape.cpp` | 122 | infershape | supporting | 输出形状推导 |
| `op_host/moe_gating_top_k_softmax_v2_tiling.cpp` | 40 | tiling入口 | core | TilingMoeGatingTopKSoftmaxV2 分发 |
| `op_host/moe_gating_top_k_softmax_v2_tiling.h` | 254 | tiling头 | core | 4 套 TilingData（EKFullLoad/KFullLoad/KRenorm/Perf）|
| `op_host/moe_gating_top_k_softmax_v2_tiling_base.cpp` | 260 | tiling | core | tiling 基类 |
| `op_host/moe_gating_top_k_softmax_v2_tiling_ek_fullload.cpp` | 281 | tiling | core | EKFullLoad tiling |
| `op_host/moe_gating_top_k_softmax_v2_tiling_k_fullload.cpp` | 210 | tiling | core | KFullLoad tiling |
| `op_host/moe_gating_top_k_softmax_v2_tiling_k_renorm.cpp` | 204 | tiling | core | KRenorm tiling |
| `op_host/moe_gating_top_k_softmax_v2_tiling_perf.cpp` | 227 | tiling | core | Perf tiling |
| `op_host/moe_gating_top_k_softmax_v2_regbase_tiling_perf.cpp` | 229 | tiling | core | arch35 regbase Perf tiling |
| `op_kernel/moe_gating_top_k_softmax_v2.cpp` | 177 | kernel入口 | core | kernel 函数（KERNEL_TYPE_MIX_AIV_1_0），~36 tiling key 分支 |
| `op_kernel/moe_gating_top_k_softmax_v2_ek_fullload.h` | 614 | kernel模板 | core | EKFullLoad 模板（expert+K 全加载）|
| `op_kernel/moe_gating_top_k_softmax_v2_k_fullload.h` | 354 | kernel模板 | core | KFullLoad 模板 |
| `op_kernel/moe_gating_top_k_softmax_v2_k_renorm.h` | 283 | kernel模板 | core | KRenorm 模板 |
| `op_kernel/moe_gating_top_k_softmax_v2_perf.h` | 840 | kernel模板 | core | Perf 高性能模板（按 colRange 分 3 档）|
| `op_kernel/arch35/moe_gating_top_k_softmax_v2_*.h` | - | kernel模板 | core | arch35 regbase 各模板 |