# moe_token_permute_grad CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `.../Triton-Ascend-KernelWiki` | source-confirmed |
| OPERATOR_NAME | `moe_token_permute_grad` | source-confirmed |
| OPERATOR_TYPE | `moe` | source-confirmed |
| VARIANTS | `backward,training` | inferred |
| mode | `static_ingest` | - |

## 2. Source file inventory

算子目录 `moe/moe_token_permute_grad/`，源码 ~2k 行（含 test）。MoE token 置换反向（unpermute = indexSelect + reshape + sum）。A2/A3 直接实现，950 转调用 MoeInitRoutingV2Grad。

完整清单见 `artifacts/moe-token-permute-grad/source_file_inventory.md`。核心文件：`README.md`、`def.cpp`（AICore config 910b/910_93，无 950）、`kernel.cpp`（调用 KernelMoeTokenUnpermute）、`moe_token_permute_grad.h`（unpermute 实现）。

## 3. Operator semantics

- 输入（2）：permutedOutputGrad/sortedIndices。
- 输出（1）：out（inputGrad）。
- 公式：indexSelect + reshape(-1, K, H) + sum(dim=1)。
- 属性：num_topk（REQUIRED，≤512）、padded_mode（OPTIONAL）。
- 类型：`moe`。

## 4. Stage / pipeline table

| Stage | Unit | Op |
|---|---|---|
| indexSelect | Vector | sortedIndices -> 间接 load permutedOutputGrad |
| reshape | Vector | [R*K, H] -> [R, K, H] |
| sum(dim=1) | Vector | 沿 K 轴 reduce -> [R, H] |
| writeback | Vector | -> out |

纯 Vector 计算，无 Cube。

## 5. Tiling parameters

- TilingKey 按 dtype：0=BF16 / 2=FP16 / 4=FP32。
- 核间：沿 token 维切分。

## 6. Core partition strategy

- 核间：沿 R 轴外切分。
- 核内：gather + sum。

## 7. Kernel templates & dispatch

- 3 种 TilingKey（dtype）。kernel 调用 `KernelMoeTokenUnpermute`（共享实现）。
- 架构：910b/910_93（arch32 直接）；950（代理转调用）。

## 8. Pipeline analysis

| Stage | CANN mechanism | Triton abstraction | Transferable? |
|---|---|---|---|
| indexSelect | sortedIndices + DataCopy | `tl.load(grad + idx)` | 是（语义） |
| reshape | view 变换 | `tl.reshape` | 是 |
| sum(dim=1) | ReduceSum | `tl.sum(x, axis=1)` | 是（数学） |
| 950 代理 | 框架转调用 | 不迁移 | 否（机制） |
| MTE 搬运 | DataCopyPad | `tl.load`/`tl.store` | 否（机制） |

## 9. CANN -> Triton-Ascend migration classification

- **transferable**：indexSelect、reshape、sum(dim=1)、unpermute 语义。
- **rewrite**：TilingKey dtype 分发、核间切分。
- **CANN-specific**：MTE DataCopyPad、950 代理转调用。
- **needs verification**：tiling 数值、UB 预算、950 代理正确性。

## 10. Triton-Ascend skeleton

`artifacts/moe-token-permute-grad/moe-token-permute-grad-triton-skeleton.py`，`reproducibility: pseudocode`。体现 indexSelect + sum。未编译、未 NPU 验证、未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `sources/ops-transformer/moe-token-permute-grad.md` | source | `ops-transformer-moe-token-permute-grad` | - | - | - |
| `wiki/kernels/moe-token-permute-grad.md` | kernel | `kernel-moe-token-permute-grad` | `[ops-transformer-moe-token-permute-grad]` | inferred | pseudocode |
| `artifacts/moe-token-permute-grad/PROVENANCE.yaml` | provenance | - | - | - | - |
| 5 个 artifacts 文件 | approach-notes | - | - | inferred | - |

## 12. Validation & retrieval

- `validate.py`: 待跑（分类器暂时不可用，但文件结构完整）。

## 13. Unverified / needs-hardware items

| 项 | 原因 |
|---|---|
| tiling 参数数值 | 依赖运行时 shape/平台 |
| UB 字节预算 | 需硬件实测 |
| 950 代理正确性 | 框架转调用，需验证参数映射 |

## 14. Modified / created files

| File | 用途 |
|---|---|
| `kernelwiki/artifacts/moe-token-permute-grad/PROVENANCE.yaml` | bundle 溯源 |
| `kernelwiki/artifacts/moe-token-permute-grad/source_file_inventory.md` | S1 |
| `kernelwiki/artifacts/moe-token-permute-grad/migration-table.md` | S3 |
| `kernelwiki/artifacts/moe-token-permute-grad/tiling_dispatch_tables.md` | S4/S5 |
| `kernelwiki/artifacts/moe-token-permute-grad/pipeline-diagram.md` | S6 |
| `kernelwiki/artifacts/moe-token-permute-grad/moe-token-permute-grad-triton-skeleton.py` | S7 |
| `kernelwiki/sources/ops-transformer/moe-token-permute-grad.md` | S8 source |
| `kernelwiki/wiki/kernels/moe-token-permute-grad.md` | S8 kernel |
| `reports/moe-token-permute-grad-ingest-report.md` | S10 |