# sparse_flash_attention - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/sparse_flash_attention/`。`fixture: true`。
注：SFA 无 arch32/arch35 子目录（单架构 arch32/A2，MLA 专用），文件直接在 `op_kernel/` 下。

| 文件（相对算子目录） | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 225 | doc | core | 功能/公式 L10-21；函数原型 L23-27；参数 L29-75；约束 L83-90 | `0502476ff6039fba...` | 算子总览 |
| `docs/aclnnSparseFlashAttention.md` | 913 | doc/api | core | aclnn API 语义 | `0f496264d160ecd1...` | 接口语义 |
| `op_host/sparse_flash_attention_def.cpp` | 101 | host/def | core | `SparseFlashAttention` OpDef：输入 L23-67；输出 L68-79；属性 L80-88；AICore config(ascend910b/910_93) L96-97 | `e38f748989b4a143...` | 算子定义 |
| `op_host/sparse_flash_attention_infershape.cpp` | 113 | host | core | infershape 规则 | `-` | shape 推导 |
| `op_host/sparse_flash_attention_tiling.cpp` | 2045 | host/tiling | core | tiling 主逻辑；InitParams 模板选择；CalcUbBmm | `e3fe448c1e9a7898...` | host tiling |
| `op_host/sparse_flash_attention_tiling.h` | 631 | host/tiling | core | `SparseFlashAttentionTilingDataMla` L196-203（5 子结构） | `fd1305d10eb477fa...` | tiling 结构体 |
| `op_kernel/sparse_flash_attention.cpp` | 53 | kernel/entry | core | `sparse_flash_attention` 模板入口 L32-54；dtype dispatch(FP16/BF16) L46-53；KERNEL_TYPE_MIX_AIC_1_2 L41 | `6a5dc0293c711867...` | kernel 入口 |
| `op_kernel/sparse_flash_attention_template_tiling_key.h` | 70 | kernel/dispatch | core | 模板参数：FLASH_DECODE/LAYOUT_T/KV_LAYOUT_T/TEMPLATE_MODE L31-37；4 合法组合 L41-69；C_TEMPLATE=0/V_TEMPLATE=1 L27-28 | `-` | 模板分发 |
| `op_kernel/sparse_flash_attention_common.h` | 201 | kernel/common | core | TEMPLATE_MODE L36；ConstInfo 同步事件 L137-142；BlockAlign L59-66 | `5549e6a85832132d...` | 公共常量 |
| `op_kernel/sparse_flash_attention_kernel_mla.h` | 1018 | kernel | core | `SparseFlashAttentionMla`；PreloadPipeline L852-894；ComputeMm1/ComputeMm2 L729-754；Process L758-775；CalcSinnerTopKBegin L941-1016；同步常量 L84-93 | `70d97224fb74c31b...` | kernel 主体（MLA） |
| `op_kernel/sparse_flash_attention_service_cube_mla.h` | 1120 | kernel | core | Cube 侧：DataCopyGmNDToL1 L47-64；DataCopyPA L72-113；InitBuffers L301-317 | `d95b92a8bb76acbf...` | Cube 服务 |
| `op_kernel/sparse_flash_attention_service_vector_mla.h` | 1464 | kernel | core | Vector 侧：MergeKv L986；GetRealS2Idx L821-831；GetKeyGmOffset L834-853；CopyInKv L896-957；CopyOutMrgeResult L960-982；InitBuffers L211-248 | `e3aad0cbed428dab...` | Vector 服务（含 MergeKv） |
| `op_host/op_api/aclnn_sparse_flash_attention.cpp` | 144 | api | core | aclnn 包装 | `-` | C++ API |
| `tests/ut/op_host/arch32/test_sparse_flash_attention_tiling.cpp` | 781 | test | indirect | tiling UT | `-` | 测试 |

注：技巧页此前的 `.../kernel_mla.h`、`.../vector_mla.h`、`.../cube_mla.h`、`.../common.h` 简记对应本表 `op_kernel/sparse_flash_attention_{kernel,service_vector,service_cube}_mla.h` 与 `op_kernel/sparse_flash_attention_common.h`。标 `-` 的 SHA-256 未单独计算。
