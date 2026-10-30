# moe_token_permute_grad - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。
算子目录：`moe/moe_token_permute_grad/`。`fixture: true`。源码 ~2k 行（含 test）。

centrality：core = 直接实现该算子；supporting = 公共/框架依赖；indirect = 测试/示例/文档。SHA-256 为前 16 位。

MoE Token Permute Grad 是 `aclnnMoeTokenPermute` 的反向传播算子。核心是 indexSelect + reshape + sum（即 unpermute）。A2/A3 训练直接实现，**950 转调用 `aclnnMoeInitRoutingV2Grad`**（代理关系）。

## docs

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 98 | doc | core | 产品支持 L5-12；公式（indexSelect+reshape+sum）L18-29；参数表 L33-79；约束（950 转调用）L83-92 | `e05c7ea6e4bb53b8…` | 算子总览 |
| `docs/aclnnMoeTokenPermuteGrad.md` | 434 | doc/api | core | aclnn API 定义 | `a98cd1fa5b86a6f9…` | 接口语义 |

## op_host

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_host/moe_token_permute_grad_def.cpp` | 51 | host/def | core | `MoeTokenPermuteGrad` OpDef；属性 num_topk（REQUIRED）L37、padded_mode（OPTIONAL）L38；AICore config（910b L46、910_93 L47，**无 950**） | `fd6e747da539d216…` | 算子定义 |
| `op_host/moe_token_permute_grad_infershape.cpp` | 46 | host | core | `InferShapeMoeTokenPermuteGrad` + `InferDataType` | `f21d92ea5d302124…` | shape/dtype 推导 |
| `op_host/moe_token_permute_grad_tiling.cpp` | 342 | host/tiling | core | tiling 主逻辑 | `287ca7b9dfe0bf4a…` | tiling 主逻辑 |
| `op_host/moe_token_permute_grad_tiling.h` | 102 | host/tiling | core | `MoeTokenPermuteGradTilingData` 结构体 | `04d79059b6543773…` | tiling 结构体 |
| `op_host/op_api/aclnn_moe_token_permute_grad.cpp/.h` | 88/34 | api | core | aclnn C++ 包装 | `6ed8a5ace052582d…`/`0159830ecb16f3a9…` | aclnn API |

## op_kernel

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_kernel/moe_token_permute_grad.cpp` | 43 | kernel/entry | core | `moe_token_permute_grad` L19-43；调用 `KernelMoeTokenUnpermute`（probs=false）；TilingKey 按 dtype（0=BF16 L30/2=FP16 L34/4=FP32 L38） | `0e823fe8e9865866…` | kernel 入口 |
| `op_kernel/moe_token_permute_grad.h` | 377 | kernel | core | `KernelMoeTokenUnpermute<T,IdxT,OutT,PROBS>` 模板类：indexSelect + reshape + sum 实现 | `209bff4956abd9b6…` | unpermute 核心实现 |

## tests

| 文件 | 行数 | role | centrality | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|
| `tests/ut/op_kernel/test_moe_token_permute_grad.cpp` | 170 | test | indirect | `d37c2b095c340f47…` | kernel UT |
| `tests/ut/op_host/test_moe_token_permute_grad_tiling.cpp` | 89 | test | indirect | `a783b649dd01fd17…` | tiling UT |
| `tests/ut/op_host/test_moe_token_permute_grad_infershape.cpp` | 82 | test | indirect | `38ce71b0c0a570ec…` | infershape UT |
| `tests/ut/op_host/op_api/test_aclnn_moe_token_permute_grad.cpp` | 59 | test | indirect | `cb5904f239fe9c60…` | aclnn UT |

注：
- 核心操作：permute 的反向 = unpermute（indexSelect + reshape(-1, numTopk, H) + sum(dim=1)）。
- kernel 实际调用 `KernelMoeTokenUnpermute`（与 `moe_token_unpermute` 共享实现），probs 始终 false。
- 三架构：A2/A3 训练直接实现（910b/910_93）；**950 转调用 `aclnnMoeInitRoutingV2Grad`**（参数映射：permutedOutputGrad->gradExpandedX、sortedIndices->expandedRowIdx、numTopk->topK、paddedMode->dropPadMode）。
- TilingKey 按 dtype：0=BF16 / 2=FP16 / 4=FP32。