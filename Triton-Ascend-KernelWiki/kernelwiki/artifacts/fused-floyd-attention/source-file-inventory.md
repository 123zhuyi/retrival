# FusedFloydAttention — 源文件清单 (S1)

来源仓库：`kernelwiki/artifacts/ops-transformer/`（CANN ops-transformer，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`，2026-07-09）。
算子目录：`attention/fused_floyd_attention/`。共 ~20 个源文件。所有路径相对算子目录。

## op_host（Host 侧 tiling/定义/inferShape）

| 文件 | 行数 | 角色 | 中心度 | 关键符号 / 职责 |
|------|------|------|--------|----------------|
| `op_host/fused_floyd_attention_def.cpp` | 95 | 定义 | core | `FusedFloydAttention` OpDef：I/O（query/key1/value1/key2/value2/atten_mask/softmax_max/softmax_sum/attention_out）、dtype FP16/BF16/FLOAT、属性 scale_value、AICore 配置 ascend910b/910_93 L20-88 |
| `op_host/fused_floyd_attention_infershape.cpp` | 93 | infershape | supporting | 输出形状推导 |
| `op_host/fused_floyd_attention_tiling.cpp` | 282 | tiling入口 | core | `TilingFusedFloydAttention` 架构分发、`TilingPrepare`、空 tensor |
| `op_host/fused_floyd_attention_tiling.h` | 365 | tiling头 | core | `FusedFloydAttentionGeneralTilingData` 结构体（inputParams/coreParams/multiCoreParams/bmm1/bmm1k1/bmm2/bmm2v2 TilingData/softmaxTilingData）、LayoutType/SparseMode 枚举 |
| `op_host/fused_floyd_attention_tiling_general.cpp` | 1363 | tiling | core | `FusedFloydAttentionTiling` 基类 + 派生类：`CalcS1S2BasicBlock`、`SetBmm1TilingInput`（4 次 matmul）、`SetMultiCoreParams`、`CalcBlockDim` |
| `op_host/fused_floyd_attention_tiling_common.cpp` | 160 | tiling公共 | supporting | 形状/dtype 校验 |
| `op_host/fused_floyd_attention_tiling_common.h` | 62 | tiling公共 | supporting | `FusedFloydAttentionCompileInfo`（aivNum/aicNum/ubSize/l1Size/l0cSize/l2CacheSize）|

## op_kernel（Kernel 侧）

| 文件 | 行数 | 角色 | 中心度 | 关键符号 / 职责 |
|------|------|------|--------|----------------|
| `op_kernel/fused_floyd_attention.cpp` | 150 | kernel入口 | core | kernel 函数 `fused_floyd_attention` L114-150；tiling key `10000000001022030943UL`（hasAttenMask）和 `10000000000022030943UL`（无 mask）；dtype FP16/BF16 |
| `op_kernel/fused_floyd_attention_s1s2_bn2gs1.h` | 1616 | kernel模板 | core | `FusedFloydAttentionS1s2Bn2gs1` 类：`Process`、`IterateBmm1`（Q·K1^T）、`IterateBmm1K1`（Q·K2^T）、`ProcessVec1`（scale + mask + softmax）、`IterateBmm2`（P·V1）、`IterateBmm2V2`（P·V2）、`ProcessVec2`（相加 + 写回） |
| `op_kernel/fused_floyd_attention_common.h` | 267 | kernel辅助 | supporting | 公共定义、ImplModeEnum、LayOutTypeEnum |

## docs / examples / tests

| 文件 | 行数 | 角色 | 中心度 | 职责 |
|------|------|------|--------|
| `README.md` | 130 | doc | core | 算子总览、FloydAttention 公式（双 QK·V）、I/O、约束 |
| `docs/FFA算子设计介绍.md` | 123 | doc | core | 计算流程设计介绍 |
| `docs/aclnnFusedFloydAttention.md` | 525 | doc | core | aclnn 接口文档 |
| `examples/test_aclnn_fused_floyd_attention.cpp` | 230 | example | indirect | 调用样例 |
| `tests/comm/fused_floyd_attention_case.cpp` | 178 | test | indirect | 测试用例 |