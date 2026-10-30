# MoeTokenUnpermuteWithEp - 源文件清单 (S1)

来源仓库：`kernelwiki/artifacts/ops-transformer/`（固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`，2026-07-09）。
算子目录：`moe/moe_token_unpermute_with_ep/`。共 ~14 个源文件。

## 核心文件

| 文件 | 行数 | 角色 | 中心度 | 职责 |
|------|------|------|--------|------|
| `README.md` | 130 | doc | core | 算子总览、前向公式（2 场景）、I/O、约束 |
| `docs/aclnnMoeTokenUnpermuteWithEp.md` | 519 | doc | core | aclnn 接口文档 |
| `op_host/moe_token_unpermute_with_ep_def.cpp` | 107 | 定义 | core | OpDef（permuted_tokens/sorted_indices/probs -> unpermuted_tokens），属性 num_topk/range/padded_mode/restore_shape，AICore 配置 ascend910b/910_93 + kirinX90/kirin9030 |
| `op_host/moe_token_unpermute_with_ep_infershape.cpp` | 60 | infershape | supporting | 输出形状推导 |
| `op_host/moe_token_unpermute_with_ep_tiling.cpp` | 407 | tiling入口 | core | TilingMoeTokenUnpermuteWithEp |
| `op_host/moe_token_unpermute_with_ep_tiling.h` | 103 | tiling头 | core | TilingData 结构体 |
| `op_kernel/moe_token_unpermute_with_ep.cpp` | 50 | kernel入口 | core | kernel 函数，4 tiling key（0:无probs/1:float/2:half/3:bf16 probs）|
| `op_kernel/moe_token_unpermute_with_ep.h` | 476 | kernel模板 | core | KernelMoeTokenUnpermuteWithEp 模板（gather + 乘 probs + reduce sum）|