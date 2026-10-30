# moe_token_permute_with_routing_map CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `.../Triton-Ascend-KernelWiki` | source-confirmed |
| OPERATOR_NAME | `moe_token_permute_with_routing_map` | source-confirmed |
| OPERATOR_TYPE | `moe` | source-confirmed |
| VARIANTS | `forward,inference,training` | inferred |
| mode | `static_ingest` | - |

## 2. Source file inventory

算子目录 `moe/moe_token_permute_with_routing_map/`，源码 ~6k 行（含 test）。routingMap 驱动的 MoE token 置换，maskedSelect + argSort + index_copy。三架构（A2/A3 + 950 arch35），arch35 含 SIMD/SIMT 二维 gather 专属模板。

完整清单见 `artifacts/moe-token-permute-with-routing-map/source_file_inventory.md`。核心文件：`README.md`（2 种模式公式）、`def.cpp`（AICore config 910b/910_93/950）、`kernel.cpp`（TilingKey 1-9/19 分发）、`masked_select_v3.h`、`moe_sort_multi_core.h`、`moe_index_copy.h`、arch35 `gather_v2_simd/simt_two_dim.h`。

## 3. Operator semantics

- 输入（3）：tokens/routing_map/probs（可选）。
- 输出（3）：permute_tokens/permute_probs/sorted_indices。
- 公式：两种模式（dropAndPad false/true），maskedSelect + argSort + index_select。
- 属性：num_out_tokens（REQUIRED）、drop_and_pad（OPTIONAL）。
- 类型：`moe`。

## 4. Stage / pipeline table

| Stage | Unit | Op |
|---|---|---|
| maskedSelect | Vector | routingMap.T -> sortedIndicesFirst |
| argSort | Vector | sortedIndicesFirst -> sortedIndicesOut（单核/多核+归并） |
| index_select | Vector | tokens.index_select(sortedIndicesOut) -> permuteTokens |
| probs 置换 | Vector | probs 同路径 -> permuteProbsOut |
| arch35 gather | Vector | Gatherv2Simd/SimtTwoDim（二维 gather 优化） |

纯 Vector 计算，无 Cube。

## 5. Tiling parameters

- TilingKey 1-9/19：1/3/5/7 单核排序、2/4/6/8 多核排序、9/19 填充模式（仅 3510）。
- 核间：多核排序 + moe_mrgsort 归并。

## 6. Core partition strategy

- 核间：沿 token 维外切分到多核排序 + 归并。
- 核内：maskedSelect + sort + index_copy。

## 7. Kernel templates & dispatch

- 13 个 kernel 头文件：masked_select_v3/moe_sort_*/moe_index_copy/moe_permute_prob/moe_mrgsort + arch35 gather。
- 架构：910b/910_93（arch32 membase）/ 950（arch35 regbase + SIMD/SIMT gather）。

## 8. Pipeline analysis

| Stage | CANN mechanism | Triton abstraction | Transferable? |
|---|---|---|---|
| maskedSelect | `KernelMaskedSelectV3` | `tl.load(tokens + mask_nonzero)` | 是（语义） |
| argSort | `MoeSort*` + `MoeMrgsort` 归并 | `tl.argsort` | 是（语义，归并算法 needs_verification） |
| index_select | `MoeindexCopyOp` | `tl.load(tokens + sorted_indices)` | 是（语义） |
| arch35 SIMD/SIMT gather | `Gatherv2Simd/SimtTwoDim` | `tl.load` 间接索引 | 是（语义，机制否） |
| 多核归并 SyncAll | CrossCoreSync | 不迁移 | 否（机制） |
| MTE 搬运 | DataCopyPad | `tl.load`/`tl.store` | 否（机制） |

## 9. CANN -> Triton-Ascend migration classification

- **transferable**：maskedSelect、argSort、index_select、probs 置换、两种模式公式。
- **rewrite**：TilingKey 分发、单核/多核排序、arch35 SIMD/SIMT gather、归并排序。
- **CANN-specific**：MTE DataCopyPad、多核归并 SyncAll、SIMD/SIMT 机制。
- **needs verification**：tiling 数值、UB 预算、归并算法等价性、SIMD/SIMT 精度。

## 10. Triton-Ascend skeleton

`artifacts/moe-token-permute-with-routing-map/moe-token-permute-with-routing-map-triton-skeleton.py`，`reproducibility: pseudocode`。体现 maskedSelect + argSort + index_select（两种模式）。未编译、未 NPU 验证、未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `sources/ops-transformer/moe-token-permute-with-routing-map.md` | source | `ops-transformer-moe-token-permute-with-routing-map` | - | - | - |
| `wiki/kernels/moe-token-permute-with-routing-map.md` | kernel | `kernel-moe-token-permute-with-routing-map` | `[ops-transformer-moe-token-permute-with-routing-map]` | inferred | pseudocode |
| `artifacts/moe-token-permute-with-routing-map/PROVENANCE.yaml` | provenance | - | - | - | - |
| 5 个 artifacts 文件 | approach-notes | - | - | inferred | - |

## 12. Validation & retrieval

- `validate.py`: 待跑（分类器暂时不可用，文件结构完整）。

## 13. Unverified / needs-hardware items

| 项 | 原因 |
|---|---|
| tiling 参数数值 | 依赖运行时 shape/平台 |
| UB 字节预算 | 需硬件实测 |
| 多核归并排序映射 | 归并算法等价性需验证 |
| arch35 SIMD/SIMT gather 精度 | 需运行时验证 |

## 14. Modified / created files

| File | 用途 |
|---|---|
| `kernelwiki/artifacts/moe-token-permute-with-routing-map/PROVENANCE.yaml` | bundle 溯源 |
| `kernelwiki/artifacts/moe-token-permute-with-routing-map/source_file_inventory.md` | S1 |
| `kernelwiki/artifacts/moe-token-permute-with-routing-map/migration-table.md` | S3 |
| `kernelwiki/artifacts/moe-token-permute-with-routing-map/tiling_dispatch_tables.md` | S4/S5 |
| `kernelwiki/artifacts/moe-token-permute-with-routing-map/pipeline-diagram.md` | S6 |
| `kernelwiki/artifacts/moe-token-permute-with-routing-map/moe-token-permute-with-routing-map-triton-skeleton.py` | S7 |
| `kernelwiki/sources/ops-transformer/moe-token-permute-with-routing-map.md` | S8 source |
| `kernelwiki/wiki/kernels/moe-token-permute-with-routing-map.md` | S8 kernel |
| `reports/moe-token-permute-with-routing-map-ingest-report.md` | S10 |