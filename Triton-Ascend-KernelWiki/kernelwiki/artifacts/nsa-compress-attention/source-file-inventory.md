# NsaCompressAttention — 源文件清单 (S1)

来源仓库：`kernelwiki/artifacts/ops-transformer/`（CANN ops-transformer，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`，2026-07-09）。
算子目录：`attention/nsa_compress_attention/`。共 ~25 个源文件。所有路径相对算子目录。

## op_host（Host 侧 tiling/定义/inferShape）

| 文件 | 行数 | 角色 | 中心度 | 关键符号 / 职责 |
|------|------|------|--------|----------------|
| `op_host/nsa_compress_attention_def.cpp` | 119 | 定义 | core | `NsaCompressAttention` OpDef：I/O dtype/layout、属性（compressBlockSize/compressStride/selectBlockSize/selectBlockCount）、AICore配置 ascend910b/910_93 L20-117 |
| `op_host/nsa_compress_attention_infershape.cpp` | 117 | infershape | supporting | 输出形状推导（attention_out、softmax_max/sum、topk_indices_out）|
| `op_host/nsa_compress_attention_tiling.cpp` | 105 | tiling入口 | core | `TilingNsaCompressAttention`分发（空 tensor、正常）、`TilingPrepareForNsaCompressAttention` |
| `op_host/nsa_compress_attention_tiling.h` | 86 | tiling头 | core | `NsaCompressAttentionGeneralTilingData`结构体（`inputParams`/`coreParams`/`multiCoreParams`/`bmm1TilingData`/`bmm2TilingData`/`softmaxTilingData`/`importanceScoreParams`/`topkTilingData`）|
| `op_host/nsa_compress_attention_tiling_common.h` | 30 | tiling公共 | supporting | `NsaCompressAttentionCompileInfo`（aivNum/aicNum/ubSize/l1Size/l0cSize/l2CacheSize）|
| `op_host/nsa_compress_attention_tiling_general.cpp` | 969 | tiling | core | `NsaCompressAttentionTilingBase`基类 + `NsaCompressAttentionVarLenScoreTiling`派生类：`CalcS1S2BasicBlock`(NSA_BASE_S1G_SIZE/gSize)、`SetBmm1/2TilingInput`、`SetMultiCoreParams`、`SetImportanceScoreParams`、`TopkTiling`、`GetWorkspaceSize` L862-964 |
| `op_host/op_api/aclnn_nsa_compress_attention.cpp` | 303 | api | core | aclnn两段式接口 `aclnnNsaCompressAttention` |
| `op_host/op_api/aclnn_nsa_compress_attention.h` | 42 | api头 | supporting | 接口声明 |
| `op_host/op_api/nsa_compress_attention.cpp` | 95 | api | supporting | 内部API注册 |
| `op_host/op_api/nsa_compress_attention.h` | 26 | api头 | supporting | 内部API声明 |

## op_kernel（Kernel 侧）

| 文件 | 行数 | 角色 | 中心度 | 关键符号 / 职责 |
|------|------|------|--------|----------------|
| `op_kernel/nsa_compress_attention.cpp` | 113 | kernel入口 | core | kernel函数 `nsa_compress_attention` L66-113；`TILING_KEY_IS(10000000000000000024UL)`分发 4×2（FP16/BF16 × hasAtten/hasTopkMask）；VarLen SameAB模板 |
| `op_kernel/nsa_compress_attention_s1s2_bn2gs1_sab.h` | 1060 | kernel模板 | core | `NsaCompressAttentionS1s2Bn2gs1SameAB`类：`Process`、`IterateBmm1`、`ProcessVec1`（softmax + mask）、`ProcessVec2`（importance score + topk）、`IterateBmm2`、`TopkCompute` |
| `op_kernel/nsa_compress_attention_var_len_score_sab.h` | 295 | kernel模板 | core | `NsaCompressAttentionVarLenScoreSameAB`类（变长 varlen 场景）|
| `op_kernel/nsa_compress_attention_common.h` | 87 | kernel辅助 | supporting | 公共定义 |

## docs / examples / tests

| 文件 | 行数 | 角色 | 中心度 | 职责 |
|------|------|------|--------|
| `README.md` | 91 | doc | core | 算子总览、NSA公式（compress attention + topk index）、I/O、约束（l/l'/d/n 16对齐，CmpSkv≤14000）|
| `docs/aclnnNsaCompressAttention.md` | 683 | doc | core | aclnn接口文档 |
| `examples/test_aclnn_nsa_compress_attention.cpp` | 235 | example | indirect | 调用样例 |
| `tests/comm/nsa_compress_attention_case.cpp` | 314 | test | indirect | 测试用例 |
| `tests/utest/ts_nsa_compress_attention.cpp` | 43 | test | indirect | 单元测试 |