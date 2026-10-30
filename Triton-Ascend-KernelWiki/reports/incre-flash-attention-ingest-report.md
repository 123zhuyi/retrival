# incre_flash_attention CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `.../Triton-Ascend-KernelWiki`（scaffold `--root` 指向 `.../kernelwiki`） | source-confirmed |
| OPERATOR_NAME | `incre_flash_attention` | source-confirmed |
| OPERATOR_TYPE | `attention` | source-confirmed（`tags.yaml#kernel_types`） |
| VARIANTS | `inference,incremental` | inferred |
| mode | `static_ingest` | - |

范围（用户确认）：V4 主语义 + V2/V3 差异对照；arch32 + arch35 详述，arch20 简述；FlashDecode/AntiQuant-MSD/PageAttention 等技术点写进 kernel 页，不新增 technique 页。

## 2. Source file inventory

算子目录 `attention/incre_flash_attention/`，源码 ~58k 行，70+ 文件，覆盖 op_host / op_kernel（arch20/32/35 + 共享）/ op_api / docs / tests / framework / op_graph。centrality 分 core/supporting/indirect。

完整清单（路径 / 行数 / role / centrality / 关键符号+行号 / SHA-256 / 关系）见 `artifacts/incre-flash-attention/source_file_inventory.md`。

核心文件：`docs/IFA算子设计介绍.md`（设计权威）、`docs/aclnnIncreFlashAttentionV4.md`（V4 主语义）、`op_host/incre_flash_attention_def.cpp`（OpDef + AICore config）、`op_host/incre_flash_attention_tiling.cpp`（arch32 tiling）/`tiling_v2.cpp`（arch35 tiling）、`op_kernel/incre_flash_attention_tilingkey.h`（arch32 dispatch）、arch32 三模板（`split_Bbn2s2_Us2`/`allvec_new`/`preload`）、arch35 `template_tiling_key.h` + `service/*` + `vector_api/vf_softmax_const.h`。

## 3. Operator semantics

- 输入（15）：query（S=1，FP16/BF16）、key/value（tensorlist，FP16/BF16/INT8）、pse_shift、atten_mask（BOOL/INT8/UINT8）、actual_seq_lengths、dequant_scale1/2、quant_scale1/2、quant_offset2、antiquant_scale/offset、block_table、kv_padding_size（V4）。
- 输出（1）：attention_out（FP16/BF16/INT8，shape 同 query）。
- 属性（6）：num_heads（必需）、scale_value、input_layout（BSH/BNSD/BSND）、num_key_value_heads（GQA）、block_size（PA）、inner_precise。
- 公式：`Softmax(scale·QK^T, atten_mask)·V`，FlashSoftmax 在线刷新，FP32 累加。
- 边界：query S1=1；B≤65536（A2/950）/ ≤256（310P）；N≤256；D≤512；PA blockSize≤512（16/32 对齐）；KV 左 padding 起点=`Smax-kvPadding-actualLen`。
- 变体：V1→V2→V3→V4（V4 新增 KV 左 padding）。
- 详见 `wiki/kernels/incre-flash-attention.md` §1-2。

## 4. Stage / pipeline table

| Stage | Entry | Unit | Op |
|---|---|---|---|
| Cube1 (Bmm1) | mm1_processor / 模板内 | AIC | Q·K^T -> attention_score（FP32，S1=1） |
| Vector1 (Vec1) | vec1_processor / 模板内 | AIV | pse add -> ×scale -> atten_mask select -> FlashSoftmax（在线 max/sum） |
| Cube2 (Bmm2) | mm2_processor / 模板内 | AIC | P·V |
| Vector2 (Vec2) | vec2_processor / 模板内 | AIV | rescale(÷softmax_sum) -> 可选 quant -> 写回 |
| FlashDecode | service_vector_flashdecode_mla / combine | AIV | split-KV 各核 Attention -> lse 加权规约 |

## 5. Tiling parameters

- 逻辑轴：B、N2（kv head）、G（q/kv 比）、S1（=1 不切）、S2。
- TilingKey：十进制 16 位组装（设计介绍 L189-244）：位0 layout / 位1 Q dtype / 位2 KV dtype / 位3 output / 位5[bit0] splitKv(FlashDecode) / 位5[bit1] PA / 位5[bit2] antiquant / 位6 antiquantMode / 位15 perfMode（模板）/ 位16 mode。
- 核内基本块：S1=1 退化（Cube Bmm1 M=1）；具体数值 `needs_verification`。

完整表见 `artifacts/incre-flash-attention/tiling_dispatch_tables.md` §B/E。

## 6. Core partition strategy

- 核间：先按 BN2 分核；`BN2 < 0.4 × 总核数` 时对 S2 外切 SplitKv 份（总块数 `BN2 × SplitKv`），即 FlashDecode。
- 负载均衡：按块负载（每块 S 实际长度）对连续块组合重排，使核间差最小，输出 blockid 数组。
- 核内：单 core 缓存有限，对 S2（或 KV 子块 S2）切分，即 FlashAttention 循环。
- 三原则：核数用满 / 各核负载均匀 / AIC-AIV 算力匹配。
- 详见 `tiling_dispatch_tables.md` §D。

## 7. Kernel templates & dispatch

- IFA 实际调用 3 种 arch32 模板：C+V 基础（`split_Bbn2s2_Us2.h`，perfMode 0）/ 全 Vector（`allvec_new.h`，perfMode 1，位于 `op_kernel/` 非 `arch32/`）/ matmul 基础 API（`preload.h`，perfMode 3）。
- `preload_dd`（perfMode 6）/ `preload_mla`（perfMode 5）设计介绍 L54/L56 标注**仅 FIA 调用，IFA 不调用**。
- 架构分发：`ascend910b`/`ascend910_93` -> arch32（`incre_flash_attention`）；`ascend950` -> arch35（`incre_flash_attention_apt`，regbase 服务化）；`ascend310p` -> arch32 入口转 arch20 解码器。
- TilingKey 十进制组装是 CANN 编译期模板注册机制，**不迁移**；Triton 用 `if constexpr`/launcher dispatch。
- 详见 `tiling_dispatch_tables.md` §A/B/C/F。

## 8. Pipeline analysis

| Stage | CANN mechanism | Purpose | Triton abstraction | Transferable? |
|---|---|---|---|---|
| Bmm1 QK | Cube mm + L0C FP32 累加 | Q·K^T（S1=1） | `tl.dot(q, k.T)` FP32 | 是（数学） |
| FlashSoftmax | 在线 max/sum/exp + PV rescale | 稳定 softmax | running max/sum + `exp(m_old-m_new)` | 是（算法） |
| Bmm2 PV | Cube mm | P·V | `tl.dot(p, v)` | 是（数学） |
| FlashDecode | split-KV + lse 加权 reduce | 长 KV 并行 | 两段 kernel（split + combine） | 是（算法） |
| AntiQuant-MSD | A 变换 int8×int8->int32 reduce | 避免大 B 反量化 | `tl.dot(int8,int32)`+reduce | 是（算法，精度 needs_verification） |
| PageAttention | 回调 GM->L1 拷贝 | KV 不连续 | `tl.load(block_table)` 间接 load | 是（语义） |
| CV 流水/N-Buffer | MTE/queue/event/Cube 双发 | 计算搬运重叠 | tile 复用/fused load-store | 否（机制） |

完整分析见 `artifacts/incre-flash-attention/pipeline-diagram.md`。

## 9. CANN -> Triton-Ascend migration classification

四桶概要：
- **transferable**：注意力公式、在线 FlashSoftmax、query S1=1 增量解码、mask select、pse/scale、FP32 累加、KV cache 变长、FlashDecode lse 规约、KV 左 padding 偏移。
- **rewrite**：三模板选择、TilingKey 分支、五轴切分、核间 BN2+SplitKv、核内 S2 切分、GQA、AntiQuant-MSD、量化/反量化、PageAttention 间接 load、分核负载均衡。
- **CANN-specific**：MTE/BufferID/queue/event/CV 手动同步、ND↔NZ、N-Buffer/Cube 双发、Cube/Vector 算力配比、TBuf 平台查询。
- **needs verification**：tiling 数值、UB/L1 字节预算、arch35 模板枚举、FlashDecode lse 精确公式、INT8 精度边界、MLA 在 IFA 支持范围、V4 在 950 支持、arch20 与 arch32 入口关系。

完整表（每条带 CANN 来源 path Lxx）见 `artifacts/incre-flash-attention/migration-table.md`。

## 10. Triton-Ascend skeleton

`artifacts/incre-flash-attention/incre-flash-attention-triton-skeleton.py`，`reproducibility: pseudocode`。体现 IFA 两招牌：增量解码（`BLOCK_M=1`）与 FlashDecode 两段式（`ifa_split_kernel` 输出 partial lse/o + `ifa_combine_kernel` lse 加权规约），含 PA/mask/pse/KV 左 padding 分支与 AntiQuant-MSD 注记。未编译、未 NPU 验证、未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `sources/ops-transformer/incre-flash-attention.md` | source | `ops-transformer-incre-flash-attention` | - | - | - |
| `wiki/kernels/incre-flash-attention.md` | kernel | `kernel-incre-flash-attention` | `[ops-transformer-incre-flash-attention]` | inferred | pseudocode |
| `artifacts/incre-flash-attention/PROVENANCE.yaml` | provenance | - | - | - | - |
| `artifacts/incre-flash-attention/source_file_inventory.md` | approach-notes | - | - | inferred | - |
| `artifacts/incre-flash-attention/migration-table.md` | approach-notes | - | - | inferred | - |
| `artifacts/incre-flash-attention/tiling_dispatch_tables.md` | approach-notes | - | - | inferred | - |
| `artifacts/incre-flash-attention/pipeline-diagram.md` | approach-notes | - | - | inferred | - |
| `artifacts/incre-flash-attention/incre-flash-attention-triton-skeleton.py` | derived-source | - | - | inferred | pseudocode |

## 12. Validation & retrieval

- `validate.py`: **OK: KernelWiki validation passed**（unique id、sources 解析、tags 词表、architectures/hardware_features/techniques/kernel_types/symptoms 均为 `tags.yaml` 值、PROVENANCE files sha256 匹配）。
- `generate-indices.py`: 生成 5 个 query 文件（by-problem/by-technique/by-hardware-feature/by-repo/by-kernel-type）。
- `query.py incre_flash_attention --compact`: 6 结果，`kernel-incre-flash-attention` + `ops-transformer-incre-flash-attention` 排前二。
- `query.py attention tiling --compact`: 10 结果，IFA kernel 命中。
- `query.py attention mask --compact`: IFA kernel 命中。
- `query.py attention triton ascend --compact`: 无 IFA hit（双关键词匹配所致，不影响检索正确性）。

## 13. Unverified / needs-hardware items

| 项 | 原因 |
|---|---|
| tiling 参数具体数值（基本块大小） | 依赖运行时 shape/平台，未硬件实测 |
| UB/L1/L0C 字节预算/占用 | `tiling.cpp` L3597-3625 workspace，需硬件实测 |
| arch35 模板枚举具体值与 ConfigValue 组合 | `template_tiling_key.h` L35-119 未逐一提取 |
| FlashDecode lse 精确公式 | 以 `vf_softmax_const.h` 为准，未逐行确认 |
| INT8/量化/AntiQuant-MSD 精度边界 | 需运行时验证 |
| MLA 在 IFA 的支持范围 | 设计介绍 L56 称 MLA 全量化模板仅 FIA 调用；`tiling.cpp` L297 有 MLA 判定，IFA 是否实际走 MLA 待确认 |
| V4 在 950PR/950DT 支持 | V4 doc L9 标 ×（KV 左 padding），def.cpp L316 注册 950，存在差异 |
| arch20（310P）与 arch32 入口关系 | 310P 经 arch32 入口转 `arch20/*` 解码器，转发条件待确认 |
| `tiling_v2.cpp` 精确行号 | 与 `tiling.cpp` 结构对应但为独立文件，本分析对 v2 用概括描述 |

## 14. Modified / created files

| File | 用途 |
|---|---|
| `kernelwiki/artifacts/incre-flash-attention/PROVENANCE.yaml` | bundle 溯源（scaffold init + 手填 upstream_repo/sha/path + update-provenance 算 files[] sha256） |
| `kernelwiki/artifacts/incre-flash-attention/source_file_inventory.md` | S1 源文件清单（70+ 文件，行数/role/centrality/关键符号+行号/SHA-256） |
| `kernelwiki/artifacts/incre-flash-attention/migration-table.md` | S3 四桶迁移分类（每条带 CANN 来源 path Lxx） |
| `kernelwiki/artifacts/incre-flash-attention/tiling_dispatch_tables.md` | S4/S5 tiling + core partition + 模板 dispatch + Triton 抽象映射 |
| `kernelwiki/artifacts/incre-flash-attention/pipeline-diagram.md` | S6 流水编排（四阶段 + 三模板流水 + FlashDecode + AntiQuant-MSD + PA） |
| `kernelwiki/artifacts/incre-flash-attention/incre-flash-attention-triton-skeleton.py` | S7 Triton skeleton（FlashDecode 两段式，pseudocode） |
| `kernelwiki/sources/ops-transformer/incre-flash-attention.md` | S8 source 页（不可变来源摘要 + 核心文件 SHA-256 + 关键代码区域） |
| `kernelwiki/wiki/kernels/incre-flash-attention.md` | S8 kernel 页（语义/流程/模板/架构/特性/Agent actions） |
| `reports/incre-flash-attention-ingest-report.md` | S10 本报告 |

## Quality checklist

- [x] 每个 `sources:` 解析到 source 页 id；每个 wiki 页引用 source。
- [x] `architectures/hardware_features/techniques/kernel_types/symptoms` 均为 `tags.yaml` 精确值。
- [x] `validate.py` 通过（无缺失 PROVENANCE、无 sha256 不匹配、无未解析 source、无未知词表）。
- [x] `performance_claims: []`（static_ingest 无 baseline/env/shapes/repeats/result）。
- [x] `confidence: inferred`；`reproducibility: pseudocode`。
- [x] 未声称 "runs / verified / benchmarked"（`validate.py` 通过仅为 schema 校验，非 kernel 运行验证）。
- [x] 未将 CANN 专有机制（MTE/BufferID/queue/event/手动同步/ND↔NZ）作为 Triton 指导。
- [x] 报告列出全部创建/编辑文件。
