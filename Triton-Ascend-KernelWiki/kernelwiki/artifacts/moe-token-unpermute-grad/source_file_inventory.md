# moe_token_unpermute_grad - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（`d28d8f9...`）。无 arch 子目录。仅 A2/A3。

| 文件 | 行数 | role | centrality | 关键符号/区域 | SHA-256 |
|---|---:|---|---|---|---|
| `README.md` | 135 | doc | core | 功能 L16-45；公式 L19-45；参数 L49-116；约束 L118-129 | `977d0415b0ccc9d1...` |
| `docs/aclnnMoeTokenUnpermuteGrad.md` | 486 | doc/api | core | aclnn API 语义 | `-` |
| `op_host/moe_token_unpermute_grad_def.cpp` | 80 | host/def | core | `MoeTokenUnpermuteGrad` OpDef：输入 L33-60；输出 L61-72；属性 L73-74；AICore config(910b/910_93) L76-77 | `bd2c0a21afcf7f5b...` |
| `op_host/moe_token_unpermute_grad_tiling.cpp` | 355 | host/tiling | core | tiling 主逻辑 | `-` |
| `op_host/moe_token_unpermute_grad_tiling.h` | 47 | host/tiling | core | tiling 类 | `-` |
| `op_kernel/moe_token_unpermute_grad.cpp` | 46 | kernel/entry | core | kernel 入口 L20-47；4 tiling key(0/1/10/11) L30-46；2 模板 ProbNone/ProbNotNone | `03e6ea18a929e265...` |
| `op_kernel/moe_token_unpermute_grad_base.h` | 197 | kernel/common | core | 反向基类 | `-` |
| `op_kernel/moe_token_unpermute_grad_prob_none.h` | 125 | kernel | core | `MoeTokenUnpermuteGradProbNone`（无 probs 反向：纯 scatter） | `-` |
| `op_kernel/moe_token_unpermute_grad_prob_not_none.h` | 373 | kernel | core | `MoeTokenUnpermuteGradProbNotNone`（有 probs 反向：scatter 加权 + probsGrad 求和） | `-` |

另含 `op_api/`(2)、`tests/`(5) 等辅助文件。