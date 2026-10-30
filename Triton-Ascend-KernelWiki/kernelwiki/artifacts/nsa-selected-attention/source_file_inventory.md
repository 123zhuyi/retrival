# nsa_selected_attention - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/nsa_selected_attention/`（训练前向）。`fixture: true`。
NSA（Native Sparse Attention）推理流水线：`nsa_compress_attention_infer`（topK 索引选择）→ `nsa_selected_attention`（根据 topKIndices gather 选中 KV 块并计算 attention，训练场景）。本算子为 NSA 训练前向的第二阶段，输出 attentionOut + softmaxMax/Sum（供反向使用）。

| 文件（相对算子目录） | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 68 | doc | core | 功能简述；参数/约束 | `92a8aff84c50cd10...` | 算子总览 |
| `docs/aclnnNsaSelectedAttention.md` | 698 | doc/api | core | 公式 L22-28；参数表 L68-353；约束 L358-380；产品支持 L5-12 | `8c41faf664756fdc...` | 接口语义（权威） |
| `op_host/nsa_selected_attention_def.cpp` | 94 | host/def | core | `NsaSelectedAttention` OpDef：输入 L19-55；输出 L56-67；属性 L68-79；AICore(910b/910_93) L88-89 | `7f0272ba20407201...` | 算子定义 |
| `op_host/nsa_selected_attention_infershape.cpp` | 107 | host | core | softmaxMax/Sum shape `[T,N1,8]` L70-80；attentionOut L82-87；仅 TND L58-62 | `c57158bda6fb6716...` | shape 推导 |
| `op_host/nsa_selected_attention_tiling.h` | 93 | host/tiling | core | `NsaSelectedAttentionTilingData` L75-83；`InputParams` L23-46；`MultiCoreParams` L50-56 | `39ba623b0e9325d2...` | tiling 结构体 |
| `op_host/nsa_selected_attention_tiling.cpp` | 346 | host/tiling | core | empty-input 路径 L56-303；tiling key 90/92/94 | `5cb40d0bd706aa55...` | tiling（空输入） |
| `op_host/nsa_selected_attention_tiling_general.cpp` | 654 | host/tiling | core | `GetTilingKey` 0/1 L565-572；`AnalyzeAttrs` L339-480；`CalcWorkspaceSize` L637-650；`GetCoreNum`（AIV）L579-593 | `a5272ae0dfba6837...` | tiling 主逻辑 |
| `op_host/op_api/aclnn_nsa_selected_attention.cpp` | 358 | api | core | `ValidateParams` L152-186；`GetWorkspaceSize` L259-346 | `0161cbcf79c54e3a...` | aclnn C++ API |
| `op_host/op_api/aclnn_nsa_selected_attention.h` | 57 | api | supporting | aclnn 声明 | `41485663aa0911c9...` | API 头 |
| `op_host/op_api/nsa_selected_attention.cpp` | 87 | api | core | l0op `NsaSelectedAttention` L29-86 | `e1411ef7b2ea779b...` | l0op 包装 |
| `op_host/op_api/nsa_selected_attention.h` | 34 | api | supporting | l0op 声明 | `4df534bc8dc4aaa1...` | API 头 |
| `op_kernel/nsa_selected_attention.cpp` | 83 | kernel/entry | core | `extern "C"` 入口 L58；`TILING_KEY_IS(0/1)` L66-79；CUBE/VECTOR 路径 L21-53 | `fc13f49e8a8be769...` | kernel 入口 |
| `op_kernel/nsa_selected_attention.h` | 750 | kernel | core | `NsaSelectedAttention` L38；`Process()` L359-392；`SelectAndGatherK` L447-481；`SelectAndGatherV` L484-521；`ProcessSoftmax` L524-581；`HandleMask` L621-666；`SetBmm1TensorA/B` L688-715 | `d669712408ea1fa2...` | kernel 主体 |
| `op_kernel/nsa_selected_attention_common.h` | 114 | kernel/common | core | `SOFTMAX_REDUCE_SIZE=8` L56；常量 L50-82；`SparseModeEnum` L85-96；`GetExtremeValue` L98-112 | `bad41d68f8526098...` | 公共常量 |
| `examples/test_aclnn_nsa_selected_attention.cpp` | 326 | test | indirect | 端到端示例 | `e0df94a69f1dc401...` | 示例 |
| `tests/utest/ts_nsa_selected_attention.cpp` | 222 | test | indirect | 端到端测试 | `af71b4fc00f3a8ec...` | 测试 |
| `tests/comm/src/nsa_selected_attention_case.cpp` | 161 | test | supporting | 测试公共用例 | `2b3e3268ce30b41c...` | 测试公共 |
| `tests/comm/src/nsa_selected_attention_param.cpp` | 99 | test | supporting | 测试参数 | `662184972e89a3c5...` | 测试公共 |
| `tests/comm/inc/nsa_selected_attention_case.h` | 59 | test | supporting | 测试公共头 | `44c47cebd94c4964...` | 测试公共 |
| `tests/comm/inc/nsa_selected_attention_param.h` | 85 | test | supporting | 测试参数头 | `7c15ca4b64782368...` | 测试公共 |

注：本算子为 NSA 按需独立实现（不共享 attn_infra 模板库）。kernel 显式分 Cube/Vector 路径（C220 架构 MIX_AIC_1_2）。`tiling_general.cpp` 为主 tiling 逻辑，`tiling.cpp` 仅处理空输入场景。