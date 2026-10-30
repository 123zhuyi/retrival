# MoeTokenPermuteWithEpGrad - 源文件清单 (S1)

来源仓库：`kernelwiki/artifacts/ops-transformer/`（固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`，2026-07-09）。
算子目录：`moe/moe_token_permute_with_ep_grad/`。共 ~13 个源文件。

## 核心文件

| 文件 | 行数 | 角色 | 中心度 | 职责 |
|------|------|------|--------|------|
| `README.md` | 116 | doc | core | 算子总览、反向公式、I/O、约束 |
| `docs/aclnnMoeTokenPermuteWithEpGrad.md` | 518 | doc | core | aclnn 接口文档 |
| `op_host/moe_token_permute_with_ep_grad_def.cpp` | 61 | 定义 | core | OpDef（permuted_tokens_output_grad/sorted_indices/permuted_probs_output_grad -> token_grad/probs_grad），属性 num_topk/range/padded_mode，AICore 配置 ascend910b/910_93 |
| `op_host/moe_token_permute_with_ep_grad_infershape.cpp` | 47 | infershape | supporting | 输出形状推导 |
| `op_host/moe_token_permute_with_ep_grad_tiling.cpp` | 407 | tiling入口 | core | TilingMoeTokenPermuteWithEpGrad |
| `op_host/moe_token_permute_with_ep_grad_tiling.h` | 104 | tiling头 | core | TilingData 结构体 |
| `op_kernel/moe_token_permute_with_ep_grad.cpp` | 60 | kernel入口 | core | kernel 函数（KERNEL_TYPE_AIV_ONLY），2 tiling key（0:无probs/1:有probs）|
| `op_kernel/moe_token_permute_with_ep_grad.h` | 476 | kernel模板 | core | KernelMoeTokenUnpermuteWithEp 模板（gather + reshape + reduce sum）|