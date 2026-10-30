# mla_prolog_v3 CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `.../Triton-Ascend-KernelWiki`（scaffold `--root` 指向 `.../kernelwiki`） | source-confirmed |
| OPERATOR_NAME | `mla_prolog_v3` | source-confirmed |
| OPERATOR_TYPE | `attention` | source-confirmed（`tags.yaml#kernel_types`） |
| VARIANTS | `prefill,inference` | inferred |
| mode | `static_ingest` | - |

## 2. Source file inventory

算子目录 `attention/mla_prolog_v3/`，源码 ~9298 行（含 test），有完整设计文档（454 行）。三架构（arch32 A2/A3 + arch35 950）。核心 kernel 实现共享 `mla_prolog/` 公共目录。

完整清单见 `artifacts/mla-prolog-v3/source_file_inventory.md`。核心文件：`docs/MlaPrologV3算子设计介绍.md`（设计权威）、`docs/aclnnMlaPrologV3WeightNz.md`（API 语义）、`op_host/mla_prolog_v3_def.cpp`（OpDef + AICore config）、`op_kernel/mla_prolog_v3.cpp`（9 模板参数入口）。

## 3. Operator semantics

- 4 路 matmul（下采样 W^{DQ}、拼接 [W^{DKV}|W^{KR}]、拼接 [W^{UQ}|W^{QR}]、上采样 W^{UK}）+ 2 RmsNorm + 2 RoPE + KVCache/KRCache 更新。
- 输出 query（q^N）、query_rope（q^R_rope）、kv_cache、kr_cache（原地更新）。
- 详见 `wiki/kernels/mla-prolog-v3.md` §1-2。

## 4. Stage / pipeline table

| Stage | Unit | Op |
|---|---|---|
| MatmulCq | Cube | x·W^{DQ} -> c^Q |
| RmsNormCq | Vector | RmsNorm(c^Q) -> c_norm^Q |
| MatmulCkvKr | Cube | x·[W^{DKV}|W^{KR}] -> [c^{KV}|k^R] |
| RmsNormCkv + RoPE k^R | Vector | RmsNorm(c^{KV}) + RoPE(k^R) -> cache |
| MatmulQcQr | Cube | c_norm^Q·[W^{UQ}|W^{QR}] -> [q^C|q^R] |
| RoPE q^R | Vector | RoPE(q^R) -> q^R_rope |
| MatmulQn | Cube | q^C·W^{UK} -> q^N |
| KVCache/KRCache update | Vector | cacheIndex scatter |

## 5. Tiling parameters

- 4 路 matmul 各不同分核策略：MatmulCq 沿 Hcq 按 AIC 核数、MatmulCkvKr 固定 64 块、MatmulQcQr 沿 Hqc 按 AIC 核数、MatmulQn 不切 N。
- TilingKey 26 位二进制（9 字段）：CACHE_MODE(0-3) / SCENARIO(4-5) / QUANT_MODE(6-9) / ENABLE_DEQUANT_OPT(10) / ENABLE_GROUP_COMPUTE_OPT(11) / EMPTY_TENSOR_MODE(12-13) / ACTUAL_SEQ_LEN_MODE(14-15) / SPLIT_M_MODE(16-17) / CV_MODE(18-25)。
- 详见 `tiling_dispatch_tables.md` §B/C。

## 6. Core partition strategy

- 核间：4 路 matmul 各按不同轴切分（Hcq/Hckv/Hqc/M），CV 之间需 6 个核间同步点。
- 核内：遍历 token block，每轮 4 matmul + 2 RmsNorm + 2 RoPE。
- 详见 `tiling_dispatch_tables.md` §C。

## 7. Kernel templates & dispatch

- 9 个 uint8_t 模板参数（`mla_prolog_v3.cpp` L25-27）对应 TilingKey 的 26 位二进制位。
- 架构分发：`ascend910b`/`ascend910_93`（arch32）/ `ascend950`（arch35，单独 dtype/layout）。
- 共享 kernel 实现：`mla_prolog/op_kernel/kernel_mla_prolog_split_n.h`（V3 与 V1/V2 共享）。

## 8. Pipeline analysis

| Stage | CANN mechanism | Triton abstraction | Transferable? |
|---|---|---|---|
| 4 matmul | Cube 高阶 API | `tl.dot` × 4 | 是（数学） |
| 2 RmsNorm | AscendC Vector API | `x * rsqrt(mean(x^2)+eps) * gamma` | 是（算法） |
| 2 RoPE | 偶奇拆分 + cos/sin 旋转 | 偶奇 split + *cos + 翻转 *sin | 是（算法） |
| KVCache scatter | DataCopyPad + cacheIndex | `tl.store(cache + idx, ...)` | 是（语义） |
| CV 6 同步点 | CrossCoreSetFlag/WaitFlag | 不迁移 | 否（机制） |
| FRACTAL_NZ | NZ 物理格式 | Triton layout（编译器） | 否（机制） |

## 9. CANN -> Triton-Ascend migration classification

- **transferable**：4 路 matmul 公式、RmsNorm、RoPE、α_q/α_kv 尺度矫正、KVCache/KRCache scatter。
- **rewrite**：TilingKey 9 字段 -> `if constexpr`、拼接 matmul `tl.concat`+split 或两次 dot、分核策略、动态/静态量化。
- **CANN-specific**：CV 6 同步点、FRACTAL_NZ、CrossCoreSetFlag/WaitFlag、All-Cube/All-Vector 全核同步。
- **needs verification**：tiling 数值、共享 kernel 实现、per-tile 量化精度、cacheIndex 更新关系、Mxfp8 支持。

完整表见 `artifacts/mla-prolog-v3/migration-table.md`。

## 10. Triton-Ascend skeleton

`artifacts/mla-prolog-v3/mla-prolog-v3-triton-skeleton.py`，`reproducibility: pseudocode`。体现 4 路 matmul + 2 RmsNorm + 2 RoPE + KVCache/KRCache scatter，含 RoPE 辅助函数 `_rope`。未编译、未 NPU 验证、未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `sources/ops-transformer/mla-prolog-v3.md` | source | `ops-transformer-mla-prolog-v3` | - | - | - |
| `wiki/kernels/mla-prolog-v3.md` | kernel | `kernel-mla-prolog-v3` | `[ops-transformer-mla-prolog-v3]` | inferred | pseudocode |
| `artifacts/mla-prolog-v3/PROVENANCE.yaml` | provenance | - | - | - | - |
| `artifacts/mla-prolog-v3/source_file_inventory.md` | approach-notes | - | - | inferred | - |
| `artifacts/mla-prolog-v3/migration-table.md` | approach-notes | - | - | inferred | - |
| `artifacts/mla-prolog-v3/tiling_dispatch_tables.md` | approach-notes | - | - | inferred | - |
| `artifacts/mla-prolog-v3/pipeline-diagram.md` | approach-notes | - | - | inferred | - |
| `artifacts/mla-prolog-v3/mla-prolog-v3-triton-skeleton.py` | derived-source | - | - | inferred | pseudocode |

## 12. Validation & retrieval

- `validate.py`: **OK: KernelWiki validation passed**。
- `generate-indices.py`: 生成 5 个 query 文件。
- `query.py mla_prolog_v3 --compact`: 2 结果，source + kernel 排前二。
- `query.py attention tiling --compact`: 无 mla hit（MLA 不涉及 flash-attention-tiling 关键词，不影响检索正确性）。

## 13. Unverified / needs-hardware items

| 项 | 原因 |
|---|---|
| tiling 参数数值 | 依赖运行时 shape/平台，未硬件实测 |
| 共享 kernel 实现（mla_prolog/） | V3 入口仅定义模板参数，核心实现以 `kernel_mla_prolog_split_n.h` 为准 |
| per-tile 量化精度边界 | 需运行时验证 |
| cacheIndex 与 cache 更新关系 | 设计介绍 L435-454，以源码为准 |
| Mxfp8 量化模式 | 需确认 Triton FP8 支持 |
| RoPE 精确公式 | 设计介绍 L361-414 优化版，与标准 RoPE 略有差异 |

## 14. Modified / created files

| File | 用途 |
|---|---|
| `kernelwiki/artifacts/mla-prolog-v3/PROVENANCE.yaml` | bundle 溯源（手动创建，scaffold 不可用） |
| `kernelwiki/artifacts/mla-prolog-v3/source_file_inventory.md` | S1 源文件清单 |
| `kernelwiki/artifacts/mla-prolog-v3/migration-table.md` | S3 四桶迁移分类 |
| `kernelwiki/artifacts/mla-prolog-v3/tiling_dispatch_tables.md` | S4/S5 tiling + dispatch |
| `kernelwiki/artifacts/mla-prolog-v3/pipeline-diagram.md` | S6 流水编排（4 matmul + 6 CV 同步点） |
| `kernelwiki/artifacts/mla-prolog-v3/mla-prolog-v3-triton-skeleton.py` | S7 Triton skeleton |
| `kernelwiki/sources/ops-transformer/mla-prolog-v3.md` | S8 source 页 |
| `kernelwiki/wiki/kernels/mla-prolog-v3.md` | S8 kernel 页 |
| `reports/mla-prolog-v3-ingest-report.md` | S10 本报告 |

## Quality checklist

- [x] 每个 `sources:` 解析到 source 页 id；每个 wiki 页引用 source。
- [x] `architectures/hardware_features/techniques/kernel_types/symptoms` 均为 `tags.yaml` 精确值。
- [x] `validate.py` 通过。
- [x] `performance_claims: []`。
- [x] `confidence: inferred`；`reproducibility: pseudocode`。
- [x] 未声称 "runs / verified / benchmarked"。
- [x] 未将 CANN 专有机制（CV 6 同步点、FRACTAL_NZ、CrossCoreSetFlag/WaitFlag）作为 Triton 指导。
- [x] 报告列出全部创建/编辑文件。