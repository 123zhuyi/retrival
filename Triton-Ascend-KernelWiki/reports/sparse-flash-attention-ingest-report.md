# sparse_flash_attention CANN->Triton-Ascend ingestion report（补完）

> 本报告记录把 `sparse_flash_attention`（SFA）入库**补完**的过程。此前 SFA 已有 source 页 + 3 个技巧页（tiling/pipeline/kv-gather，由历史会话产出），但缺 kernel 页与 artifact bundle。本次补齐 kernel 页 + bundle，并把 source 页从"最小锚点"升级为含 SHA-256 的完整来源页。

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `kernelwiki/` | source-confirmed |
| OPERATOR_NAME | `sparse_flash_attention` | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred（源自 `ops-transformer/attention/`） |
| VARIANTS | inference（sparse KV + MLA-absorb + FlashDecoding + PA） | inferred |
| commit | `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` | source-confirmed |
| mode | static_ingest（补完） | - |

## 2. Source file inventory

算子目录 `attention/sparse_flash_attention/`，~18 文件，**无 arch32/arch35 子目录**（单架构 A2/A3，MLA 专用）。核心：`README.md`(225L)、`docs/aclnnSparseFlashAttention.md`(913L)、`op_host/sparse_flash_attention_def.cpp`(101L)、`op_host/sparse_flash_attention_tiling.cpp`(2045L)、`op_kernel/sparse_flash_attention.cpp`(53L)、`op_kernel/sparse_flash_attention_template_tiling_key.h`(70L)、`op_kernel/sparse_flash_attention_kernel_mla.h`(1018L)、`op_kernel/sparse_flash_attention_service_{cube,vector}_mla.h`(1120/1464L)、`op_kernel/sparse_flash_attention_common.h`(201L)。

完整清单（含 SHA-256）见 `artifacts/sparse-flash-attention/source_file_inventory.md`。技巧页此前的 `.../kernel_mla.h` 等简记已解析为真实路径（`op_kernel/sparse_flash_attention_kernel_mla.h` 等）。

## 3. Operator semantics

- **功能**：大序列推理稀疏注意力，sparse_indices 驱动离散 KV 取数，"只计算关键部分"（README L12-21）。
- **公式**：`softmax(Q·K̃^T/√dk)·Ṽ`，K̃/Ṽ 为稀疏选择的重要 K/V（README L16-20）。
- **输入**：query/key/value（必选，FP16/BF16，D=512，N1∈{1,2,4,8,16,32,64,128}，KV_N=1）、sparse_indices（必选，INT32）、block_table/actual_seq_lengths_query/_kv/query_rope/key_rope（可选，rope D=64）（def.cpp L23-67）。
- **输出**：attention_out（必选）；softmax_max/softmax_sum（return_softmax_lse=True，FLOAT）（def.cpp L68-79）。
- **属性**：scale_value(必选)、sparse_block_size(默认1，[1,128] 2的幂)、layout_query/layout_kv(默认BSND)、sparse_mode(默认3=rightDownCausal)、attention_mode(仅2=MLA-absorb)、return_softmax_lse(默认false)（def.cpp L80-88）。
- **约束**：仅推理；D(nope)=512+D(rope)=64=576；sparse_block_size 整除 block_size；block_size 16 倍数≤1024；TND+BSND 不支持（README L83-90）。
- **MLA-absorb**（attention_mode=2）：Q/K 沿 D 拼 nope+rope=576，key/value 共享张量（README L73）。

## 4. Stage / pipeline table

5 阶段 CV 流水（V_TEMPLATE 含 MergeKv 前置，详见 [[technique-sparse-attention-pipeline]]）：

| Stage | Unit | Entry | Op |
|---|---|---|---|
| Stage0 MergeKv | AIV | service_vector_mla.h:986 | 离散 KV 聚合到连续 buffer |
| Stage1 Bmm1 | AIC | kernel_mla.h:729 | Q·K^T |
| Stage2 Vec1 | AIV | service_vector_mla.h | FlashSoftmax（在线 max/sum） |
| Stage3 Bmm2 | AIC | kernel_mla.h:743 | P·V |
| Stage4 Vec2 | AIV | service_vector_mla.h | rescale + 写回 |

## 5-8. tiling / dispatch / pipeline

详见 artifact `tiling_dispatch_tables.md` 与既有技巧页 [[technique-sparse-attention-tiling]]、[[technique-sparse-attention-pipeline]]、[[technique-sparse-kv-gather]]。要点：2 模板（C/V_TEMPLATE，`sparseBlockSize≤4 && s2Size≠0` -> V）；基本块 mBaseSize=128/s2BaseSize 动态/PRELOAD_NUM=2；B/N2/G/S1 四轴分核 + FlashDecoding S2 核间切分；MLA D=576；AIC:AIV=1:2；5 个 CV sync event（CANN 专有）。

## 9. Migration classification

完整四分类表见 `artifacts/sparse-flash-attention/migration-table.md`。概要：
- **transferable**：稀疏注意力公式、在线 FlashSoftmax、MLA-absorb D 拼接、sparse_mode mask、scale、FP32 累加、softmax_max/sum 输出。
- **rewrite**：sparse_indices 驱动离散 KV 取数、MergeKv 聚合、PA blocktable、基本块->constexpr、core id->program_id、FlashDecoding S2 切分、C/V 模板。
- **CANN-specific**：MTE 手动搬运、CrossCore sync event、PreloadPipeline 3-slot、ND↔NZ、MIX_AIC_1_2 配比。
- **needs_verification**：tiling 最优值、UB 预算、FlashDecoding 合并数值稳定性、PA offset 算术。

## 10. Triton-Ascend skeleton

见 `artifacts/sparse-flash-attention/sparse-flash-attention-triton-skeleton.py`。`reproducibility: pseudocode`。表达：sparse_indices 驱动 S2 离散 gather、MergeKv 抽象为 gather->连续 tensor、MLA nope+rope D 拼接、在线 softmax、FlashDecoding 跨核 reduce、可选 PA/return_lse。未编译/未验证/未 benchmark。

## 11. KB pages written（本次补完）

| Path | type | id | 状态 |
|---|---|---|---|
| `wiki/kernels/sparse-flash-attention.md` | kernel | `kernel-sparse-flash-attention` | **新建** |
| `artifacts/sparse-flash-attention/PROVENANCE.yaml` | provenance | - | scaffold 生成 + update-provenance 冻结 |
| `…/source_file_inventory.md` | approach-notes | - | 新建 |
| `…/tiling_dispatch_tables.md` | approach-notes | - | 新建 |
| `…/migration-table.md` | approach-notes | - | 新建 |
| `…/sparse-flash-attention-triton-skeleton.py` | derived-source | - | 新建 |
| `sources/ops-transformer/sparse-flash-attention.md` | source | `ops-transformer-sparse-flash-attention` | **升级**（加 SHA-256 + 衍生引用） |

既有（未改）：`wiki/techniques/sparse-attention-{tiling,pipeline}.md`、`wiki/techniques/sparse-kv-gather.md`。kernel 页 `related` 引用 `kernel-flash-attention-score`、`kernel-prompt-flash-attention` 及 3 个 sparse 技巧页。

## 12. Validation & retrieval

```
$ python kernelwiki/scripts/validate.py
OK: KernelWiki validation passed

$ python kernelwiki/scripts/generate-indices.py   -> OK

$ python kernelwiki/scripts/query.py sparse_flash_attention --compact
# 3 hits：source 页 + kernel-sparse-flash-attention(新) + kernel-flash-attention-score ✓

$ python kernelwiki/scripts/query.py attention sparse --compact
# 10 hits：kernel-sparse-flash-attention + 3 sparse 技巧页 + source 页 + FA 系列 ✓
```

PROVENANCE 4 文件 sha256 已冻结；validate 通过。

## 13. Unverified / needs-hardware items

| 项 | 原因 |
|---|---|
| tiling 参数最优值（s2BaseSize 等） | 依赖运行时 shape/平台 |
| UB/L1 字节预算 | 需硬件实测 |
| FlashDecoding S2 核间切分合并数值稳定性 | 多核独立 softmax 后合并，需验证 |
| PA offset 算术 | 需张量实例确认 |
| sparse_block_size 与 block_size 整除边界 | 运行时验证 |
| V_TEMPLATE 触发阈值（sparseBlockSize≤4） | 来源源码，但最优性需验证 |

## 14. Modified / created files

**新建：**
- `kernelwiki/wiki/kernels/sparse-flash-attention.md` - kernel 知识页
- `kernelwiki/artifacts/sparse-flash-attention/{PROVENANCE.yaml, source_file_inventory.md, tiling_dispatch_tables.md, migration-table.md, sparse-flash-attention-triton-skeleton.py}` - artifact bundle
- `reports/sparse-flash-attention-ingest-report.md` - 本报告

**修改：**
- `kernelwiki/sources/ops-transformer/sparse-flash-attention.md` - 升级为含 SHA-256 的完整来源页
- `kernelwiki/queries/by-*.md` - `generate-indices.py` 重新生成

## Quality checklist

- [x] `sources:` 解析到真实 source 页；kernel 页 + 3 技巧页均引用 source。
- [x] `architectures/hardware_features/techniques/kernel_types/symptoms` 均为 `tags.yaml` 词表值。
- [x] `validate.py` 通过。
- [x] `performance_claims: []`。
- [x] `confidence: inferred`、`reproducibility: pseudocode`。
- [x] 无"已运行/已验证/benchmarked"声明。
- [x] CANN 专有机制未作为 Triton 指南，已抽象。
- [x] 报告列出全部新建/修改文件。
