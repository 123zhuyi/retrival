# kv_quant_sparse_flash_attention CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `.../Triton-Ascend-KernelWiki` | source-confirmed |
| OPERATOR_NAME | `kv_quant_sparse_flash_attention` | source-confirmed |
| OPERATOR_TYPE | `attention` | source-confirmed（`tags.yaml#kernel_types`） |
| VARIANTS | `inference` | inferred |
| mode | `static_ingest` | - |

## 2. Source file inventory

算子目录 `attention/kv_quant_sparse_flash_attention/`，源码 ~10k 行。仅推理场景（A2/A3 推理 + 950 推理），无训练。核心是 Sparse FlashAttention + Per-Token-Head-Tile-128 KV 量化 + MLA-absorb 模式。arch32 和 arch35 各有独立的 `kernel_mla.h` + `service_cube_mla.h` + `service_vector_mla.h`。

完整清单见 `artifacts/kv-quant-sparse-flash-attention/source_file_inventory.md`。核心文件：`README.md`（参数/约束）、`def.cpp`（AICore config）、`template_tiling_key.h`（4 字段模板参数）、`kernel_mla.h`（MLA-absorb kernel 核心）、`service_vector_mla.h`（Vector 服务）。

## 3. Operator semantics

- 输入：query（BF16/FP16，D=576 nope+rope）、key/value（INT8 或 FP8_E4M3FN/HIFLOAT8，D=656 nope+rope+dequant_scale）、sparse_indices（INT32 离散索引）、scale_value、key/value_quant_mode（per-tile=2）、block_table 等。
- 输出：attention_out（BF16/FP16）。
- 公式：`softmax((Q · Dequant(K)^T) / √d) · Dequant(V)`。
- 约束：D=576/656，KV_N=1，sparse_block_size 整除 block_size，量化参数和数据混合存放（quant_scale_repo_mode=1）。
- 详见 `wiki/kernels/kv-quant-sparse-flash-attention.md` §1-2。

## 4. Stage / pipeline table

| Stage | Unit | Op |
|---|---|---|
| sparse KV gather | Vector | sparse_indices → 离散 load KV（INT8/FP8） |
| KV dequant per-tile | Vector | cast → * dequant_scale |
| Bmm1 QK (MLA) | Cube | Q·K^T（nope+rope 分段 dot） |
| mask + scale | Vector | sparse_mode=3 rightDownCausal |
| FlashSoftmax | Vector | 在线 max/sum/exp rescale |
| Bmm2 PV | Cube | P·V |
| rescale + writeback | Vector | ÷softmax_sum → output |

## 5. Tiling parameters

- 4 模板参数：FLASH_DECODE(bool) / LAYOUT_T(BSND/TND) / KV_LAYOUT_T(BSND/TND/PA_BSND) / TEMPLATE_MODE(C/V)。
- 5 种合法组合（`template_tiling_key.h` L42-55）。
- 核间：沿 B/N2/S1 轴外切分核。
- 详见 `tiling_dispatch_tables.md` §B/C。

## 6. Core partition strategy

- 核间：沿 B/N2/S1 轴外切分核，按 aicNum 分配。
- 核内：sparse_indices 驱动离散取数 → KV dequant → Bmm1 → mask → FlashSoftmax → Bmm2 → rescale。

## 7. Kernel templates & dispatch

- 4 模板参数（`template_tiling_key.h` L31-37）。arch35（950）分支处理 FP8_E4M3FN/HIFLOAT8。
- 架构分发：`ascend910b`/`ascend910_93`（arch32 INT8）/ `ascend950`（arch35 FP8）。

## 8. Pipeline analysis

| Stage | CANN mechanism | Triton abstraction | Transferable? |
|---|---|---|---|
| sparse KV gather | sparse_indices + DataCopy | `tl.load(indices)` → 间接 `tl.load` | 是（语义） |
| KV dequant per-tile | Cast + Mul + tile 128 | `cast * scale` | 是（语义） |
| Bmm1 QK (MLA) | Cube MLA-absorb | `tl.dot` 分段 | 是（数学） |
| FlashSoftmax | 在线 max/sum/exp | running softmax | 是（算法） |
| Bmm2 PV | Cube | `tl.dot` | 是（数学） |
| MTE DataCopy | `DataCopyPad` | `tl.load` | 否（机制） |

## 9. CANN -> Triton-Ascend migration classification

- **transferable**：注意力公式、sparse_indices 离散取数、KV dequant per-tile、MLA-absorb 分段 dot、sparse_mode mask、FlashSoftmax。
- **rewrite**：4 模板参数 -> `if constexpr`、per-tile 反量化、PA 间接 load、FP8 cast。
- **CANN-specific**：MTE DataCopyPad、Cube/Vector 手动同步、C_TEMPLATE/V_TEMPLATE 模板选择机制。
- **needs verification**：tiling 数值、per-tile 量化精度、sparse_indices 有效值约定、FP8 精度边界。

## 10. Triton-Ascend skeleton

`artifacts/kv-quant-sparse-flash-attention/kv-quant-sparse-flash-attention-triton-skeleton.py`，`reproducibility: pseudocode`。体现 sparse_indices 离散取数 + MLA-absorb 分段 dot + KV dequant per-tile。未编译、未 NPU 验证、未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `sources/ops-transformer/kv-quant-sparse-flash-attention.md` | source | `ops-transformer-kv-quant-sparse-flash-attention` | - | - | - |
| `wiki/kernels/kv-quant-sparse-flash-attention.md` | kernel | `kernel-kv-quant-sparse-flash-attention` | `[ops-transformer-kv-quant-sparse-flash-attention]` | inferred | pseudocode |
| `artifacts/kv-quant-sparse-flash-attention/PROVENANCE.yaml` | provenance | - | - | - | - |
| `artifacts/kv-quant-sparse-flash-attention/source_file_inventory.md` | approach-notes | - | - | inferred | - |
| `artifacts/kv-quant-sparse-flash-attention/migration-table.md` | approach-notes | - | - | inferred | - |
| `artifacts/kv-quant-sparse-flash-attention/tiling_dispatch_tables.md` | approach-notes | - | - | inferred | - |
| `artifacts/kv-quant-sparse-flash-attention/pipeline-diagram.md` | approach-notes | - | - | inferred | - |
| `artifacts/kv-quant-sparse-flash-attention/kv-quant-sparse-flash-attention-triton-skeleton.py` | derived-source | - | - | inferred | pseudocode |

## 12. Validation & retrieval

- `validate.py`: **OK: KernelWiki validation passed**。
- `generate-indices.py`: 待补跑（分类器暂时不可用）。
- `query.py`: 待补跑。

## 13. Unverified / needs-hardware items

| 项 | 原因 |
|---|---|
| tiling 参数数值 | 依赖运行时 shape/平台，未硬件实测 |
| per-tile 量化精度边界（key_quant_mode=2） | 需运行时验证 |
| sparse_indices 每行有效值在前半部分 | 约定而非强制 |
| FP8_E4M3FN/HIFLOAT8 精度边界（arch35） | 需运行时验证 |
| quant_scale_repo_mode=1 精确 scale 提取偏移 | 以源码为准 |

## 14. Modified / created files

| File | 用途 |
|---|---|
| `kernelwiki/artifacts/kv-quant-sparse-flash-attention/PROVENANCE.yaml` | bundle 溯源（手动创建） |
| `kernelwiki/artifacts/kv-quant-sparse-flash-attention/source_file_inventory.md` | S1 源文件清单 |
| `kernelwiki/artifacts/kv-quant-sparse-flash-attention/migration-table.md` | S3 四桶迁移分类 |
| `kernelwiki/artifacts/kv-quant-sparse-flash-attention/tiling_dispatch_tables.md` | S4/S5 tiling + dispatch |
| `kernelwiki/artifacts/kv-quant-sparse-flash-attention/pipeline-diagram.md` | S6 流水编排 |
| `kernelwiki/artifacts/kv-quant-sparse-flash-attention/kv-quant-sparse-flash-attention-triton-skeleton.py` | S7 Triton skeleton |
| `kernelwiki/sources/ops-transformer/kv-quant-sparse-flash-attention.md` | S8 source 页 |
| `kernelwiki/wiki/kernels/kv-quant-sparse-flash-attention.md` | S8 kernel 页 |
| `reports/kv-quant-sparse-flash-attention-ingest-report.md` | S10 本报告 |