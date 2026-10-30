# fused_infer_attention_score CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `.../Triton-Ascend-KernelWiki`（scaffold `--root` 指向 `.../kernelwiki`） | source-confirmed |
| OPERATOR_NAME | `fused_infer_attention_score` | source-confirmed |
| OPERATOR_TYPE | `attention` | source-confirmed（`tags.yaml#kernel_types`） |
| VARIANTS | `inference,prefill,decode` | inferred |
| mode | `static_ingest` | - |

范围（用户确认）：V5 主语义（950/arch35，最新最全）+ V4 跨架构差异（A2/A3）；arch32 + arch35 详述（无 310P）；深入分析 `attn_infra/` 分层架构；技术点写进 kernel 页，不新增 technique 页。

## 2. Source file inventory

算子目录 `attention/fused_infer_attention_score/`，源码 ~39k 行 + `attn_infra/` 52 个 `.hpp`（~14k 行）。覆盖 op_host（顶层 + arch32/ + arch35/ + op_api/）/ op_kernel（顶层 + attn_infra/）/ docs / tests / examples / op_graph。FIA 无 `算子设计介绍.md`。

完整清单见 `artifacts/fused-infer-attention-score/source_file_inventory.md`。核心：V5 aclnn doc（主语义）、def.cpp（AICore config）、tiling.cpp + arch32/tiling_v3 + arch35/tiling_v2、`attn_infra/`（gemm/epilogue/layout/arch 分层）、flash_attention_regular.h/regular_decode.h（prefill/decode 模板）、_template_tiling_key.h（3116 行模板分发）。

## 3. Operator semantics

- 统一 prefill（Q_S>1）+ decode（Q_S=1）。
- 输入（V5，30+）：query/key/value（FP16/BF16/INT8/HIFLOAT8/FLOAT8_E4M3FN；KV 还 INT4/FLOAT4_E2M1）、pse_shift、atten_mask、actual_seq_lengths/actual_seq_lengths_kv、deq/quant scale（5 个）、block_table、query/kv_padding_size、key/value_antiquant_scale/offset（分离反量化）、key/value_shared_prefix、query/key_rope、learnable_sink、q/kv_start_idx 等。
- 输出：attention_out、softmax_lse（softmaxLseFlag）。
- 属性（15）：num_heads、scale_value、pre/next_tokens、input_layout、num_key_value_heads、sparse_mode、inner_precise、block_size、antiquant_mode、softmax_lse_flag、key/value_antiquant_mode、query_quant_mode、pse_type。
- 公式：`Softmax(scale·QK^T, atten_mask)·V`，FlashSoftmax 在线刷新，FP32 累加。
- 变体：V1-V5（V5 新增 qStartIdx/kvStartIdx/pseType、FP8/FP4、分离 KV 反量化、sharedPrefix、RoPE、learnableSink、softmaxLse）。
- 详见 `wiki/kernels/fused-infer-attention-score.md` §1-2。

## 4. Stage / pipeline table

| Stage | Entry（attn_infra） | Unit | Op |
|---|---|---|---|
| Cube1 (Bmm1) | `gemm/block/block_mmad_qk` | AIC | Q·K^T -> attention_score（FP32） |
| Vector1 (Vec1) | `epilogue/tile_common` + `epilogue/block/block_epilogue_online_softmax` | AIV | pse/scale/mask + FlashSoftmax |
| Cube2 (Bmm2) | `gemm/block/block_mmad_pv` | AIC | P·V |
| Vector2 (Vec2) | `epilogue/block/block_epilogue_rescale_o` | AIV | rescale(÷softmax_sum) + 写回；可选 softmaxLse |
| FlashDecode | `regular_decode` + combine | AIV | split-KV + lse 加权规约（仅 decode） |

## 5. Tiling parameters

- 逻辑轴：B、N2、G、S1、S2；prefill S1>1 参与切分，decode S1=1 不切。
- arch32 tilingkey（`tilingkey.h` 259 行）：十进制组装（layout/dtype/splitKv/PA/antiquant/perfMode/mode），具体字段 `needs_verification`（无设计文档）。
- arch35 模板参数化（`template_tiling_key.h` 3116 行）：`if constexpr` 分发，具体枚举 `needs_verification`。
- 详见 `tiling_dispatch_tables.md` §B/C。

## 6. Core partition strategy

- 核间：BN2 分核；decode + `BN2<0.4×核数` 时 S2 外切 SplitKv（FlashDecode）。
- prefill：S1 参与切分（多 token）。
- 核内：S2 分块 FlashAttention 循环。
- 详见 `tiling_dispatch_tables.md` §D。

## 7. Kernel templates & dispatch

- prefill 模板 `flash_attention_regular.h`（1059 行）；decode 模板 `flash_attention_regular_decode.h`（745 行，含 FlashDecode）。
- arch35 用 `attn_infra/` 分层 + `_template_tiling_key.h`（3116 行）`if constexpr`；arch32 用 `tilingkey.h` 十进制 + regular/regular_decode。
- 架构分发：`ascend910b`/`ascend910_93` -> arch32（`fused_infer_attention_score`）；`ascend950` -> arch35（`fused_infer_attention_score_apt`）；无 310P。
- tilingkey 十进制是 CANN 编译期注册机制，**不迁移**。
- 详见 `tiling_dispatch_tables.md` §A/F。

## 8. Pipeline analysis

核心：`attn_infra/` 分层（gemm=matmul，epilogue=softmax/rescale，layout=ND/NZ，arch=buffer/sync）对应 QK->softmax->PV->rescale。四阶段 gemm+epilogue 交替；prefill/decode 双模板；FlashDecode（decode）；V5 分离 KV 反量化/sharedPrefix/RoPE/learnableSink/softmaxLse；layout/arch/gemm Cube 搬运为 CANN 专有，不迁移。

完整分析见 `artifacts/fused-infer-attention-score/pipeline-diagram.md`。

## 9. CANN -> Triton-Ascend migration classification

四桶概要：
- **transferable**：注意力公式、在线 FlashSoftmax、统一 prefill+decode、mask、pse（pseType）、scale、FP32 累加、KV cache、变长 actualSeqLen、FlashDecode lse 规约、softmaxLse、query/kv padding、q/kv_start_idx。
- **rewrite**：attn_infra 分层映射（gemm->tl.dot，epilogue->online softmax）、tilingkey 分支、模板分发（regular/regular_decode）、arch32/35、sharedPrefix、RoPE、learnableSink、分离 KV 反量化、queryQuantMode、量化、PageAttention、MLA。
- **CANN-specific**：layout ND↔NZ、arch buffer/sync、gemm Cube L1/L0 物理搬运、MTE/BufferID/queue/event/CV 手动同步。
- **needs verification**：tiling 数值/行号、arch35 模板枚举、attn_infra 精确类名/行号、V5 在 A2/A3 支持、FP8/FP4/INT4 精度、长序列超时阈值。

完整表见 `artifacts/fused-infer-attention-score/migration-table.md`。

## 10. Triton-Ascend skeleton

`artifacts/fused-infer-attention-score/fused-infer-attention-score-triton-skeleton.py`，`reproducibility: pseudocode`。体现 FIA 统一 prefill/decode（`BLOCK_M` 由 Q_S 决定）+ FlashDecode 两段式（split + combine），含 PA/sharedPrefix/RoPE 分支与 attn_infra 分层映射注记。未编译、未 NPU 验证、未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `sources/ops-transformer/fused-infer-attention-score.md` | source | `ops-transformer-fused-infer-attention-score` | - | - | - |
| `wiki/kernels/fused-infer-attention-score.md` | kernel | `kernel-fused-infer-attention-score` | `[ops-transformer-fused-infer-attention-score]` | inferred | pseudocode |
| `artifacts/fused-infer-attention-score/PROVENANCE.yaml` | provenance | - | - | - | - |
| `artifacts/fused-infer-attention-score/source_file_inventory.md` | approach-notes | - | - | inferred | - |
| `artifacts/fused-infer-attention-score/migration-table.md` | approach-notes | - | - | inferred | - |
| `artifacts/fused-infer-attention-score/tiling_dispatch_tables.md` | approach-notes | - | - | inferred | - |
| `artifacts/fused-infer-attention-score/pipeline-diagram.md` | approach-notes | - | - | inferred | - |
| `artifacts/fused-infer-attention-score/fused-infer-attention-score-triton-skeleton.py` | derived-source | - | - | inferred | pseudocode |

## 12. Validation & retrieval

- `validate.py`: **OK: KernelWiki validation passed**。
- `generate-indices.py`: 生成 5 个 query 文件。
- `query.py fused_infer_attention_score --compact`: 2 结果，`kernel-fused-infer-attention-score` + source 页排前二。
- `query.py attention tiling --compact`: FIA kernel 命中。
- `query.py attention mask --compact`: FIA kernel 命中。

## 13. Unverified / needs-hardware items

| 项 | 原因 |
|---|---|
| FIA 无 `算子设计介绍.md` | tiling/pipeline 知识仅来自源码与 aclnn doc，无设计文档交叉验证 |
| `tiling.cpp` 精确行号 | 静态读取，关键逻辑行号可能偏移 |
| tiling 参数具体数值 | 依赖运行时 shape/平台，未硬件实测 |
| arch35 模板枚举与 ConfigValue | `template_tiling_key.h`（3116 行）未逐一提取 |
| arch32 tilingkey 字段表 | `tilingkey.h`（259 行）未逐一提取（无设计文档） |
| `attn_infra` 精确类名/行号 | 关键符号由文件名+分层推断，以源码为准 |
| UB/L1/L0C 字节预算 | `tiling.cpp` workspace L1536-1537，需硬件实测 |
| V5 在 A2/A3 支持范围 | V5 doc L10-15 标仅 950，def.cpp 注册 910b/910_93 |
| FP8/FP4/INT4 精度边界 | 需运行时验证 |
| FlashDecode/softmaxLse 精确公式 | 以 `block_epilogue_online_softmax.hpp` 为准 |
| 长序列超时阈值 | README L100-154，依赖 B/S/N/D 乘积 |

## 14. Modified / created files

| File | 用途 |
|---|---|
| `kernelwiki/artifacts/fused-infer-attention-score/PROVENANCE.yaml` | bundle 溯源（scaffold + 手填 upstream + update-provenance） |
| `kernelwiki/artifacts/fused-infer-attention-score/source_file_inventory.md` | S1 源文件清单（含 attn_infra 52 个 .hpp 分层） |
| `kernelwiki/artifacts/fused-infer-attention-score/migration-table.md` | S3 四桶迁移分类 |
| `kernelwiki/artifacts/fused-infer-attention-score/tiling_dispatch_tables.md` | S4/S5 tiling + 模板 dispatch + attn_infra Triton 映射 |
| `kernelwiki/artifacts/fused-infer-attention-score/pipeline-diagram.md` | S6 流水编排（attn_infra 分层 + prefill/decode + FlashDecode + V5 特性） |
| `kernelwiki/artifacts/fused-infer-attention-score/fused-infer-attention-score-triton-skeleton.py` | S7 Triton skeleton（统一 prefill/decode + FlashDecode，pseudocode） |
| `kernelwiki/sources/ops-transformer/fused-infer-attention-score.md` | S8 source 页（19 个核心文件完整 SHA-256） |
| `kernelwiki/wiki/kernels/fused-infer-attention-score.md` | S8 kernel 页（语义/attn_infra 分层/模板/架构/V5 特性/Agent actions） |
| `reports/fused-infer-attention-score-ingest-report.md` | S10 本报告 |

## Quality checklist

- [x] 每个 `sources:` 解析到 source 页 id；每个 wiki 页引用 source。
- [x] `architectures/hardware_features/techniques/kernel_types/symptoms` 均为 `tags.yaml` 精确值。
- [x] `validate.py` 通过。
- [x] `performance_claims: []`。
- [x] `confidence: inferred`；`reproducibility: pseudocode`。
- [x] 未声称 "runs / verified / benchmarked"（`validate.py` 通过仅为 schema 校验）。
- [x] 未将 CANN 专有机制（attn_infra layout/arch、MTE/BufferID/queue/event/Cube 物理搬运）作为 Triton 指导。
- [x] 报告列出全部创建/编辑文件。
