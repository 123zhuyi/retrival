# nsa_compress_attention_infer - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/nsa_compress_attention_infer/`。`fixture: true`。
NSA（Native Sparse Attention）家族：`nsa_compress`、`nsa_compress_attention`、`nsa_compress_attention_infer`、`nsa_compress_grad`、`nsa_compress_with_cache`、`nsa_selected_attention`、`nsa_selected_attention_grad`、`nsa_selected_attention_infer`。本算子为 NSA 推理流水线的第一阶段——压缩注意力计算 + 重要性分数 + topK 索引选择。

| 文件（相对算子目录） | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 125 | doc | core | NSA 简述；功能/参数/约束 | `42962f28106b19bf...` | 算子总览 |
| `docs/aclnnNsaCompressAttentionInfer.md` | 651 | doc/api | core | NSA 公式 L22-28；参数表 L77-353；约束 L436-442；产品支持 L5-12 | `6a933afd5b6a36d2...` | 接口语义（权威） |
| `op_host/nsa_compress_attention_infer_def.cpp` | 97 | host/def | core | `NsaCompressAttentionInfer` OpDef：输入 L19-56；输出 L57-63；属性 L64-86；AICore(910b/910_93) L88-89 | `d5b1a47b5a878cc3...` | 算子定义 |
| `op_host/nsa_compress_attention_infer_infershape.cpp` | 88 | host | core | output=query 形状拷贝 L61-62；topKOutput 推导 L69-82 | `60b83533c53f9896...` | shape 推导 |
| `op_host/nsa_compress_attention_infer_tiling.h` | 243 | host/tiling | core | `NsaCompressAttentionInferTilingData` L22-68；`NsaCompressAttentionInferTiling` L80；`NCAICompileInfo` L71 | `b04a9b1b63190339...` | tiling 结构体 |
| `op_host/nsa_compress_attention_infer_tiling_utils.h` | 72 | host/tiling | supporting | 辅助函数（最大 head dim 计算等） | `ba57c61553711bf4...` | tiling 工具 |
| `op_host/nsa_compress_attention_infer_tiling.cpp` | 675 | host/tiling | core | `Tiling` 主逻辑 L457-558；`AssignQSeqLen` L313-326；`AssignTask` L327-445；`CalcWorkspaceSize` L452-455 | `d5f04c1f900fbb48...` | host tiling 主逻辑 |
| `op_host/op_api/aclnn_nsa_compress_attention_infer.cpp` | 336 | api | core | `ValidateParams` L108-251；`GetWorkspaceSize` L260-310 | `a61659aafeb25a99...` | aclnn C++ API |
| `op_host/op_api/aclnn_nsa_compress_attention_infer.h` | 94 | api | supporting | aclnn 声明 | `c56107d6ec070157...` | API 头 |
| `op_host/op_api/nsa_compress_attention_infer.cpp` | 105 | api | core | l0op `NsaCompressAttentionInfer` L29-99 | `999b8199ea69f5d9...` | l0op 包装 |
| `op_host/op_api/nsa_compress_attention_infer.h` | 40 | api | supporting | l0op 声明 | `3adb1bcea48e0c65...` | API 头 |
| `op_kernel/nsa_compress_attention_infer.cpp` | 55 | kernel/entry | core | `extern "C"` 入口 L22-26；`KERNEL_MIX_AIC_1_2` L32；2 tiling key 分派 L35-54 | `6518c087057ad9f8...` | kernel 入口 |
| `op_kernel/nsa_compress_attention_infer.h` | 1590 | kernel | core | `NsaCompressAttentionInfer` 调度器 L1567-1588；`NsaCompressAttentionInferAic` cube 路径 L40-689；`NsaCompressAttentionInferAiv` vector 路径 L690-1565；`ProcessMm1` L517-559；`ProcessMm2` L562-608；`ProcessSoftmax` L1083-1116；`ProcessImportanceScore` L1258-1273；`ProcessTopK` L1479-1526 | `3419e3da635d722d...` | kernel 主体 |
| `op_kernel/ncai_public_define.h` | 55 | kernel/common | core | `NCAIType` L46-52；同步事件 id MM1_READY=0/SOFTMAX_READY=1/MM2_READY=2 L34-36；常量 L27-39 | `ce95206e21126d8b...` | 公共常量 |
| `examples/test_aclnn_nsa_compress_attention_infer.cpp` | 218 | test | indirect | 端到端示例 | `921d9e729b18f2e2...` | 示例 |
| `tests/utest/op_host/test_nsa_compress_attention_infer_tiling.cpp` | 191 | test | indirect | tiling UT | `2835a6ccc58ff7d9...` | 单测 |
| `tests/utest/op_kernel/test_nsa_compress_attention_infer_kernel.cpp` | 191 | test | indirect | kernel UT | `912807fe28e1e177...` | 单测 |
| `tests/comm/ncai_case.cpp` | 257 | test | supporting | 测试公共用例 | `147bee5e8bfb23b6...` | 测试公共 |
| `tests/comm/ncai_case.h` | 107 | test | supporting | 测试公共头 | `0c3c22380d2b4349...` | 测试公共 |
| `tests/comm/ts_ncai.h` | 22 | test | supporting | 测试辅助 | `44bb02a41fc39c12...` | 测试公共 |
| `op_host/CMakeLists.txt` | 29 | build | indirect | host 构建 | - | 构建 |
| `CMakeLists.txt` | 20 | build | indirect | 算子构建 | `2ca29331c55d06bd...` | 构建 |
| `tests/CMakeLists.txt` | 17 | build | indirect | 测试构建 | - | 构建 |

注：本算子为 NSA 家族按需独立实现，不共享其他 NSA 算子的 `attn_infra` 模板库（与 block_sparse_attention 系列不同）。kernel 路径显式分 Cube/Vector 两套实现（C220 架构 MIX_AIC_1_2）。`tests/CMakeLists.txt` 未单独计算 SHA-256。`tiling_utils.h` 为辅助工具。