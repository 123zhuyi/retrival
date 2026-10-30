---
id: ops-transformer-moe-token-permute-with-ep-grad
title: CANN ops-transformer MoeTokenPermuteWithEpGrad source snapshot
type: source
repo: cann/ops-transformer
path: moe/moe_token_permute_with_ep_grad/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-31
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeTokenPermuteWithEpGrad 来源摘要

仓库快照固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（2026-07-09）。`MoeTokenPermuteWithEpGrad` 是 `MoeTokenPermuteWithEp` 的反向传播算子。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `moe/moe_token_permute_with_ep_grad/`

## 2. 核心源文件清单与 SHA-256

| 源文件 | SHA-256 | 作用 |
|------|------|------|
| `README.md` | `6bb17fb1fc1830a9a5cfdfef6f823b4ec45b752a4aac6dfe8ae811d7f0f574c2` | 算子总览、反向公式、约束 |
| `op_host/moe_token_permute_with_ep_grad_def.cpp` | `807ccce9191171c7325701ee009141b2904191913f9a2e4860bde03b6fced190` | OpDef（AICore 配置 ascend910b/910_93）|
| `op_host/moe_token_permute_with_ep_grad_tiling.h` | `e704cad5fc1e4ee4a74b677369c0bd11586212d7fb0e4a457e91bb4694baf392` | TilingData 结构体 |
| `op_kernel/moe_token_permute_with_ep_grad.cpp` | `9881a3eb55618030ff3046020393e7769167ea9f81abd00d44aeb949a998b8f5` | kernel 入口（2 tiling key）|
| `op_kernel/moe_token_permute_with_ep_grad.h` | `36b4abbc957372a24e1a6d055f9de3ee2e572ae5a6a24e819ac2ebfeb3012efa` | KernelMoeTokenUnpermuteWithEp 模板 |

## 3. 关键代码区域摘要

### 反向公式（README.md）
1. `sortedIndices = sortedIndices[range[0] <= i < range[1]]` - range 过滤
2. `tokenGradOut = permutedTokensOutputGrad.indexSelect(0, sortedIndices)` - gather
3. `tokenGradOut = tokenGradOut.reshape(-1, topK, hiddenSize)` - reshape
4. `tokenGradOut = tokenGradOut.sum(dim=1)` - 跨 topK reduce sum

### 输入输出（def.cpp）
- 输入：permuted_tokens_output_grad（BF16/FP16/FLOAT）、sorted_indices（INT32）、permuted_probs_output_grad（可选）
- 输出：token_grad、probs_grad（BF16/FP16/FLOAT）
- 属性：num_topk（必需）、range（[0,0] 默认）、padded_mode（false）

### 架构（def.cpp L56-57）
仅 ascend910b/910_93（不支持 ascend950）

### Kernel dispatch（kernel L53-59）
`KERNEL_TYPE_AIV_ONLY`。2 tiling key：0（无 probs）/ 1（有 probs）。1 模板类 `KernelMoeTokenUnpermuteWithEp`。

### 约束（README.md L108-110）
top_k <= 512；不支持 paddedMode=True；rangeOptional 为空时回退到 `MoeTokenPermuteGrad`

## 4. 衍生引用
- `wiki/kernels/moe-token-permute-with-ep-grad.md`
- `artifacts/moe-token-permute-with-ep-grad/`