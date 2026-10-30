# MoeFinalizeRoutingV2Grad - 源文件清单 (S1)

来源仓库：`kernelwiki/artifacts/ops-transformer/`（固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`，2026-07-09）。
算子目录：`moe/moe_finalize_routing_v2_grad/`。共 ~30 个源文件。

## 核心文件

| 文件 | 行数 | 角色 | 中心度 | 职责 |
|------|------|------|--------|------|
| `README.md` | 174 | doc | core | 算子总览、3 场景反向公式、I/O |
| `docs/aclnnMoeFinalizeRoutingV2Grad.md` | 547 | doc | core | aclnn 接口文档 |
| `op_host/moe_finalize_routing_v2_grad_def.cpp` | 153 | 定义 | core | OpDef（grad_y/expanded_row_idx 必选 + expanded_x/scales/expert_idx/bias 可选 -> grad_expanded_x/grad_scales），2 套 AICore 配置 |
| `op_host/moe_finalize_routing_v2_grad_infershape.cpp` | 120 | infershape | supporting | 输出形状推导 |
| `op_host/moe_finalize_routing_v2_grad_tiling.cpp` | 574 | tiling入口 | core | TilingMoeFinalizeRoutingV2Grad 分发 |
| `op_host/moe_finalize_routing_v2_grad_tiling.h` | 199 | tiling头 | core | TilingData 结构体 |
| `op_host/moe_finalize_routing_v2_grad_tiling_membase.cpp` | 108 | tiling | core | membase tiling |
| `op_host/moe_finalize_routing_v2_grad_split_h_tiling_arch35.cpp` | 164 | tiling | core | arch35 SplitH tiling |
| `op_host/moe_finalize_routing_v2_grad_not_split_h_tiling_arch35.cpp` | 142 | tiling | core | arch35 NotSplitH tiling |
| `op_host/moe_finalize_routing_v2_grad_without_scale_tiling_arch35.cpp` | 97 | tiling | core | arch35 WithoutScale tiling |
| `op_kernel/moe_finalize_routing_v2_grad.cpp` | 58 | kernel入口 | core | kernel 函数（KERNEL_TYPE_MIX_AIV_1_0），4 tiling key 分支 |
| `op_kernel/moe_finalize_routing_v2_grad_base.h` | 442 | kernel模板 | core | 基础模板 |
| `op_kernel/moe_finalize_routing_v2_grad_with_scale_cut_h.h` | 133 | kernel模板 | core | WithScale CutH |
| `op_kernel/moe_finalize_routing_v2_grad_with_scale_not_cut_h.h` | 113 | kernel模板 | core | WithScale NotCutH |
| `op_kernel/moe_finalize_routing_v2_grad_without_scale_cut_h.h` | 121 | kernel模板 | core | WithoutScale CutH |
| `op_kernel/moe_finalize_routing_v2_grad_without_scale_not_cut_h.h` | 111 | kernel模板 | core | WithoutScale NotCutH |
| `op_kernel/arch35/moe_finalize_routing_v2_grad_base.h` | 442 | kernel模板 | core | arch35 基础模板 |
| `op_kernel/arch35/moe_finalize_routing_v2_grad_regbase.h` | 212 | kernel模板 | core | arch35 regbase 入口 |
| `op_kernel/arch35/moe_finalize_routing_v2_grad_regbase_cut_h.h` | 616 | kernel模板 | core | arch35 regbase CutH |
| `op_kernel/arch35/moe_finalize_routing_v2_grad_regbase_not_cut_h.h` | 554 | kernel模板 | core | arch35 regbase NotCutH |