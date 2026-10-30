# nsa_compress_grad - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/nsa_compress_grad/`。`fixture: true`。源码 ~1939 行。

centrality：core = 直接实现该算子；supporting = 公共/框架依赖；indirect = 测试/示例/文档。SHA-256 为前 16 位。

NSA compress grad 是 NSA（Native Sparse Attention）压缩算子的反向计算。仅支持 Atlas A2 训练（ascend910b / ascend910_93），仅 TND layout。

## docs

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 57 | doc | core | 产品支持（仅 A2 训练）L5-9；公式 L14-25；参数表 L28-39 | `a1b2c3d4e5f6a7b8…` | 算子总览 |
| `docs/aclnnNsaCompressGrad.md` | 535 | doc/api | core | aclnn API 定义（outputGrad/input/weight → inputGrad/weightGrad）；参数表 L59-206；约束 L289-293 | `c5d6e7f8a9b0c1d2…` | 接口语义 |

## op_host

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_host/nsa_compress_grad_def.cpp` | 78 | host/def | core | `NsaCompressGrad` OpDef：输入输出 dtype（FP16/BF16）L22-59；属性（compressBlockSize/Stride/actSeqLenType/layoutOptional）L46-49；AICore config（ascend910b L72、ascend910_93 L73） | `d4e5f6a7b8c9d0e1…` | 算子定义 |
| `op_host/nsa_compress_grad_tiling.cpp` | 241 | host/tiling | core | `NsaCompressGradTiling`：`DoOpTiling` L87-136（分核/属性读取）；`GetWorkspaceSize` L142-150（workspace = 16MB + blockSize*headNum*aivNum*4 + tSeqLen*headNum*headDim*4 + 1MB）；`PostTiling` L157-179（SetBlockDim=aivNum）；`CheckTilingData` L218-236（16 对齐校验）；`GetTilingKey`=0 L152-155 | `e6f7a8b9c0d1e2f3…` | tiling 主逻辑 |
| `op_host/nsa_compress_grad_tiling.h` | 113 | host/tiling | core | `NsaCompressGradTiling` 类定义与 `NsaCompressGradTilingData` 结构体（batchSize/numOfBlock/numOfHead/dimOfHead/blockSize/blockStride/seqLenType/headToProcess/headRemainder/ubMaxSize） | `f7a8b9c0d1e2f3a4…` | tiling 数据结构 |
| `op_host/nsa_compress_grad_infershape.cpp` | 48 | host | core | `InferShapeNsaCompressGrad` + `InferDataTypeNsaCompressGrad` | `a8b9c0d1e2f3a4b5…` | shape/dtype 推导 |

## op_kernel

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_kernel/nsa_compress_grad.cpp` | 57 | kernel/entry | core | kernel 入口：`NsaCompressGrad` 函数 L22-57（调用 `NsaCompressGradND<T>::Init()` + `Process()` + `MoveResult()`） | `b9c0d1e2f3a4b5c6…` | kernel 入口 |
| `op_kernel/nsa_compress_grad.h` | 565 | kernel | core | `NsaCompressGradND<T>` 模板类：`Init` L35-53 / `InitTilingData` L56-89 / `Process` L235-301（主循环：遍历 block → head → row，CopyIn → ComputeWtGrad → CopyOutWtGrad → ComputeInputGrad → CopyOutInputGrad）/ `ComputeWtGrad` L330-349（Mul+BlockReduceSum+WholeReduceSum）/ `ComputeInputGrad` L352-370（Cast+BroadCast+Mul+CastByCondition）/ `CopyInCmpGrad` L304-318 / `CopyInInput` L320-327 / `CopyInWt` L393-399 / `CopyOutWtGrad` L373-390（AtomicAdd）/ `CopyOutInputGrad` L402-416 / `MoveResult` L419-437 / `DeterministicComputeSumWtGrad` L440-500 / `MoveWtGrad` L503-519 / `MoveInputGrad` L522-562 | `c0d1e2f3a4b5c6d7…` | kernel 核心实现 |
| `op_kernel/nsa_compress_grad_base.h` | 179 | kernel | core | `NsaCompressGradND` 辅助函数：`GetSeqLenTarBch` / `CompressBlkNum` / `CastByCondition` / `PresetFlag` / `ClearPresetFlag` / `CeilDiv` 等工具 | `d1e2f3a4b5c6d7e8…` | 基础工具 |
| `op_kernel/nsa_compress_grad_common.h` | 46 | kernel | supporting | 公共类型/宏定义 | `e2f3a4b5c6d7e8f9…` | 公共定义 |

## tests / examples

| 文件 | 行数 | role | centrality | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|
| `tests/comm/nsa_compress_grad_case.cpp` | 226 | test | indirect | `f3a4b5c6d7e8f9a0…` | 测试用例实现 |
| `tests/comm/nsa_compress_grad_case.h` | 18 | test | indirect | `a4b5c6d7e8f9a0b1…` | 测试用例头 |
| `tests/utest/ts_nsa_compress_grad.h` | 32 | test | indirect | `b5c6d7e8f9a0b1c2…` | UT 头 |
| `tests/utest/ts_nsa_compress_grad_tc.cpp` | 65 | test | indirect | `c6d7e8f9a0b1c2d3…` | UT 实现 |
| `examples/test_aclnn_nsa_compress_grad.cpp` | 62 | example | indirect | `d7e8f9a0b1c2d3e4…` | aclnn 调用示例 |

注：
- NSA compress grad 是当前分析过的 attention 算子中最小的（~1939 行），仅 A2 训练架构（ascend910b/910_93），无 arch35/arch20，无 PagedAttention、FlashDecode、MLA 等复杂特性。
- 核心计算为两个 matmul 反向（dw = dk_cmp · K^T，dk = W^T · dk_cmp），对应 CANN kernel 中的 `ComputeWtGrad`（Mul+BlockReduceSum）与 `ComputeInputGrad`（BroadCast+Mul）。
- TilingKey 恒为 0（单模板），分核按 aivNum 沿 block 轴外切。
- workspace 含 sysWorkspace（16MB）+ weightGradAtomic + inputGradAtomic（overlap 场景）+ 1MB 缓冲。
- SHA-256 为示意值（前 16 位），完整值需在固定提交下重算。