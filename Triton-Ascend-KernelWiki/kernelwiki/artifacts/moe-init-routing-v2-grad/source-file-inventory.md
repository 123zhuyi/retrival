# MoeInitRoutingV2Grad - 源文件清单 (S1)

来源仓库：`kernelwiki/artifacts/ops-transformer/`（固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`，2026-07-09）。
算子目录：`moe/moe_init_routing_v2_grad/`。共 ~30 个源文件。

## 核心文件

| 文件 | 行数 | 角色 | 中心度 | 职责 |
|------|------|------|--------|------|
| `README.md` | 88 | doc | core | 算子总览、反向公式、I/O |
| `op_host/moe_init_routing_v2_grad_def.cpp` | 59 | 定义 | core | OpDef（grad_expanded_x/expanded_row_idx -> grad_x），属性 top_k/drop_pad_mode/active_num，AICore 配置 ascend910b/910_93/950 |
| `op_host/moe_init_routing_v2_grad_tiling.h` | 140 | tiling头 | core | TilingData（coreNum/n/e/c/cols/k/activeNum + MoeV2GradComputeParams）、3 套 regbase TilingData（SplitH 200xxx/FullLoad 300xxx/通用 400xxx）|
| `op_host/moe_init_routing_v2_grad_tiling.cpp` | 231 | tiling入口 | core | TilingMoeInitRoutingV2Grad 分发 |
| `op_host/moe_init_routing_v2_grad_tiling_base.cpp` | 262 | tiling | core | membase tiling 基类 |
| `op_host/moe_init_routing_v2_grad_tiling_arch35.cpp` | 155 | tiling | core | arch35 regbase tiling |
| `op_host/moe_init_routing_v2_grad_full_load_tiling_arch35.cpp` | 163 | tiling | core | arch35 FullLoad tiling |
| `op_host/moe_init_routing_v2_grad_split_h_tiling_arch35.cpp` | 154 | tiling | core | arch35 SplitH tiling |
| `op_kernel/moe_init_routing_v2_grad.cpp` | 85 | kernel入口 | core | kernel 函数（KERNEL_TYPE_AIV_ONLY），9 tiling key 分支 |
| `op_kernel/moe_init_routing_v2_grad_base.h` | 214 | kernel模板 | core | 基础模板 |
| `op_kernel/moe_init_routing_v2_grad_with_dropless.h` | 208 | kernel模板 | core | Dropless 模式（无 drop）|
| `op_kernel/moe_init_routing_v2_grad_with_activate.h` | 218 | kernel模板 | core | Activate 模式 |
| `op_kernel/moe_init_routing_v2_grad_with_pos_drop_and_pad_zero.h` | 218 | kernel模板 | core | Position+Drop+PadZero 模式 |
| `op_kernel/arch35/moe_init_routing_v2_grad.h` | 220 | kernel模板 | core | arch35 regbase 入口 |
| `op_kernel/arch35/moe_init_routing_v2_grad_base.h` | 365 | kernel模板 | core | arch35 基础模板 |
| `op_kernel/arch35/moe_init_routing_v2_grad_full_load.h` | 164 | kernel模板 | core | arch35 FullLoad |
| `op_kernel/arch35/moe_init_routing_v2_grad_split_h.h` | 173 | kernel模板 | core | arch35 SplitH |