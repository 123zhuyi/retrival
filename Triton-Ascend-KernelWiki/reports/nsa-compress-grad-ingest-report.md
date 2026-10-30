# nsa_compress_grad CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `.../Triton-Ascend-KernelWiki`（scaffold `--root` 指向 `.../kernelwiki`） | source-confirmed |
| OPERATOR_NAME | `nsa_compress_grad` | source-confirmed |
| OPERATOR_TYPE | `attention` | source-confirmed（`tags.yaml#kernel_types`） |
| VARIANTS | `backward` | inferred |
| mode | `static_ingest` | - |

## 2. Source file inventory

算子目录 `attention/nsa_compress_grad/`，源码 ~1939 行，当前分析过的 attention 算子中最小的。仅 A2 训练架构（ascend910b/910_93），仅 TND layout，TilingKey=0 单模板。覆盖 op_host / op_kernel / docs / tests / examples。

完整清单见 `artifacts/nsa-compress-grad/source_file_inventory.md`。核心文件：`nsa_compress_grad.h`（565 行，NsaCompressGradND 模板类）、`nsa_compress_grad_tiling.cpp`（241 行，tiling 主逻辑）、`nsa_compress_grad_base.h`（179 行，辅助工具）。

## 3. Operator semantics

- 输入（4）：outputGrad（dk_cmp，[T,N,D]）、input（K，[T,N,D]）、weight（W，[blockSize,N]）、actSeqLen（可选，cumsum）。
- 输出（2）：inputGrad（dk，[T,N,D]）、weightGrad（dw，[blockSize,N]）。
- 属性（4）：compressBlockSize（必需/16 对齐）、compressStride（必需/16 对齐）、actSeqLenType（0=cumsum）、layoutOptional（TND）。
- 公式：`dw = dk_cmp · K^T`（weight 梯度 reduce），`dk = W^T · dk_cmp`（input 梯度 broadcast+mul）。
- 详见 `wiki/kernels/nsa-compress-grad.md` §1-2。

## 4. Stage / pipeline table

| Stage | Entry（CANN） | Unit | Op |
|---|---|---|---|
| CopyInCmpGrad | `CopyInCmpGrad` L304-318 | MTE+Vector | GM→UB: DataCopy + Cast(FP16→FP32) |
| CopyInInput | `CopyInInput` L320-327 | MTE | GM→UB: DataCopy input |
| CopyInWt | `CopyInWt` L393-399 | MTE | GM→UB: DataCopy weight |
| ComputeWtGrad | `ComputeWtGrad` L330-349 | Vector | Mul + BlockReduceSum + WholeReduceSum → dw |
| CopyOutWtGrad | `CopyOutWtGrad` L373-390 | MTE | AtomicAdd UB→GM |
| ComputeInputGrad | `ComputeInputGrad` L352-370 | Vector | Cast + BroadCast + Mul → dk |
| CopyOutInputGrad | `CopyOutInputGrad` L402-416 | MTE | AtomicAdd（overlap）或 DataCopy |
| 核间规约 | `MoveResult` L419-437 | Vector+MTE | 确定性规约 / 单核搬运 / ping-pong 搬运 |

## 5. Tiling parameters

- 分核：block 轴外切，`nBlockPerCore = totalBlockNum / aivNum`，剩余块分配给前 nRemainder 个核。
- TilingKey 恒为 0（单模板 `NsaCompressGradND<T>`）。
- 核内：block→head→row 三层循环，`nHeadOnce_` 由 UB 容量决定，`nRowsOnce_` = `maxHeadNumInUb >= headNum ? maxHeadNumInUb/headNum : 1`。
- 详见 `tiling_dispatch_tables.md` §C/D。

## 6. Core partition strategy

- 核间：block 维外切（`totalBlockNum / coreNum`），`SetBlockDim(aivNum)`。
- 核内：按 head→row 遍历，每行 7 步流水（CopyInCmpGrad→CopyInInput→CopyInWt→ComputeWtGrad→CopyOutWtGrad→ComputeInputGrad→CopyOutInputGrad）。
- 详见 `tiling_dispatch_tables.md` §C。

## 7. Kernel templates & dispatch

- 单模板 `NsaCompressGradND<T>`（`nsa_compress_grad.h`，565 行），TilingKey 恒为 0。
- 架构分发：`ascend910b`/`ascend910_93` -> `nsa_compress_grad`（arch32），无 arch35/arch20。
- 属性 `deterministicFlag_` 分支（`CopyOutWtGrad` L376-388）与 `isOverLap_` 分支（`CopyOutInputGrad` L405-409）。

## 8. Pipeline analysis

| Stage | CANN mechanism | Purpose | Triton abstraction | Transferable? |
|---|---|---|---|---|
| dw = dk_cmp · K^T | Mul + BlockReduceSum + WholeReduceSum | weight 梯度 matmul reduce | `tl.dot(dk_cmp, K.T)` | 是（数学） |
| dk = W^T · dk_cmp | Cast + BroadCast + Mul | input 梯度 broadcast mul | `tl.dot(W.T, dk_cmp)` | 是（数学） |
| AtomicAdd | `SetAtomicAdd` + `DataCopyPad` | 多核写同位置 | `tl.atomic_add` | 是（语义） |
| overlap 累加 | AtomicAdd workspace + 搬运 | blockSize > blockStride | `tl.atomic_add` 或 double buffer | 是（语义） |
| DataCopy GM→UB | MTE `DataCopyPad` + SetFlag/WaitFlag | 数据搬运 | `tl.load` | 否（机制） |
| 确定性规约 | `DeterministicComputeSumWtGrad` | 树形规约 | Triton kernel launch barrier | 否（机制） |

## 9. CANN -> Triton-Ascend migration classification

四桶概要：
- **transferable**：dw = dk_cmp · K^T、dk = W^T · dk_cmp、TND 变长 cumsum、compressBlockSize/Stride 对齐约束、滑窗切分逻辑。
- **rewrite**：AtomicAdd weightGrad、overlap inputGrad AtomicAdd、BlockReduceSum/WholeReduceSum → `tl.sum`、BroadCast → `tl.broadcast_to`、block 维外切分核、TilingKey=0 单模板。
- **CANN-specific**：MTE DataCopyPad/SetFlag/WaitFlag 同步、workspace 清零 ping-pong buffer、确定性树形规约。
- **needs verification**：compressBlockSize/Stride 最优值、UB 字节预算、actSeqLenType=0 cumsum 精确语义、AtomicAdd 多核精度边界。

完整表见 `artifacts/nsa-compress-grad/migration-table.md`。

## 10. Triton-Ascend skeleton

`artifacts/nsa-compress-grad/nsa-compress-grad-triton-skeleton.py`，`reproducibility: pseudocode`。体现 `tl.dot(dk_cmp, K.T)`（weight 梯度）+ `tl.dot(W.T, dk_cmp)`（input 梯度）+ `tl.atomic_add` 规约。未编译、未 NPU 验证、未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `sources/ops-transformer/nsa-compress-grad.md` | source | `ops-transformer-nsa-compress-grad` | - | - | - |
| `wiki/kernels/nsa-compress-grad.md` | kernel | `kernel-nsa-compress-grad` | `[ops-transformer-nsa-compress-grad]` | inferred | pseudocode |
| `artifacts/nsa-compress-grad/PROVENANCE.yaml` | provenance | - | - | - | - |
| `artifacts/nsa-compress-grad/source_file_inventory.md` | approach-notes | - | - | inferred | - |
| `artifacts/nsa-compress-grad/migration-table.md` | approach-notes | - | - | inferred | - |
| `artifacts/nsa-compress-grad/tiling_dispatch_tables.md` | approach-notes | - | - | inferred | - |
| `artifacts/nsa-compress-grad/pipeline-diagram.md` | approach-notes | - | - | inferred | - |
| `artifacts/nsa-compress-grad/nsa-compress-grad-triton-skeleton.py` | derived-source | - | - | inferred | pseudocode |

## 12. Validation & retrieval

- `validate.py`: **OK: KernelWiki validation passed**。
- `generate-indices.py`: 生成 5 个 query 文件。
- `query.py nsa_compress_grad --compact`: 2 结果，kernel + source 排前二。
- `query.py attention tiling --compact`: 无 nsa hit（关键词不匹配，NSA 不涉及 tiling 关键词，不影响检索正确性）。

## 13. Unverified / needs-hardware items

| 项 | 原因 |
|---|---|
| compressBlockSize/Stride 最优值 | 依赖运行时 shape/平台，未硬件实测 |
| UB 字节预算 | `nsa_compress_grad.h` L118-156（InitBuffers），需硬件实测 |
| actSeqLenType=0 cumsum 精确语义 | 以 `GetSeqLenTarBch` 源码为准 |
| AtomicAdd 多核精度边界 | 需运行时验证 |
| deterministicFlag 与确定性规约算法 | 以 `DeterministicComputeSumWtGrad` 源码为准 |

## 14. Modified / created files

| File | 用途 |
|---|---|
| `kernelwiki/artifacts/nsa-compress-grad/PROVENANCE.yaml` | bundle 溯源 |
| `kernelwiki/artifacts/nsa-compress-grad/source_file_inventory.md` | S1 源文件清单 |
| `kernelwiki/artifacts/nsa-compress-grad/migration-table.md` | S3 四桶迁移分类 |
| `kernelwiki/artifacts/nsa-compress-grad/tiling_dispatch_tables.md` | S4/S5 tiling + dispatch |
| `kernelwiki/artifacts/nsa-compress-grad/pipeline-diagram.md` | S6 流水编排 |
| `kernelwiki/artifacts/nsa-compress-grad/nsa-compress-grad-triton-skeleton.py` | S7 Triton skeleton |
| `kernelwiki/sources/ops-transformer/nsa-compress-grad.md` | S8 source 页 |
| `kernelwiki/wiki/kernels/nsa-compress-grad.md` | S8 kernel 页 |
| `reports/nsa-compress-grad-ingest-report.md` | S10 本报告 |

## Quality checklist

- [x] 每个 `sources:` 解析到 source 页 id；每个 wiki 页引用 source。
- [x] `architectures/hardware_features/techniques/kernel_types/symptoms` 均为 `tags.yaml` 精确值。
- [x] `validate.py` 通过。
- [x] `performance_claims: []`。
- [x] `confidence: inferred`；`reproducibility: pseudocode`。
- [x] 未声称 "runs / verified / benchmarked"。
- [x] 未将 CANN 专有机制（MTE DataCopyPad/SetFlag-WaitFlag/workspace ping-pong/确定性规约）作为 Triton 指导。
- [x] 报告列出全部创建/编辑文件。