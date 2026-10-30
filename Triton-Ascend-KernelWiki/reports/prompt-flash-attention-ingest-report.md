# prompt_flash_attention CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `kernelwiki/` | source-confirmed |
| OPERATOR_NAME | `prompt_flash_attention` | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred（源自 `ops-transformer/attention/` 目录） |
| VARIANTS | 全部（inference/prefill 为主；KV-cache/增量/PA/quant/MLA 为扩展） | inferred |
| commit | `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory

算子目录 `attention/prompt_flash_attention/`，含 ~100 文件（op_host/op_kernel/arch32/arch35/docs/tests/examples）。核心文件：`README.md`(174L)、`docs/PFA算子设计介绍.md`(137L)、`docs/aclnnPromptFlashAttention{,V2,V3}.md`、`op_host/prompt_flash_attention_def.cpp`(211L)、`op_host/prompt_flash_attention_tiling.cpp`(6520L)、`op_host/prompt_flash_attention_tiling_v2.cpp`(5234L)、`op_kernel/prompt_flash_attention.cpp`(58L)、`op_kernel/prompt_flash_attention_tilingkey.h`(153L)、`op_kernel/arch35/prompt_flash_attention_template_tiling_key{,_enum}.h`、`op_kernel/arch35/service/{mm1,vector1,mm2,vector2}.h`、`op_kernel/arch35/vf_softmaxflashv3.h`(2703L)。

完整清单（path / role / centrality / 关键符号 / 行号区域 / SHA-256 / 与算子关系）见 `artifacts/prompt-flash-attention/source_file_inventory.md`。

## 3. Operator semantics

- **功能**：全量推理（prefill）FlashAttention，支持 sparse、actualSeqLengthsKv 变长、INT8/FP8 量化、高精度/高性能模式（README L16）。
- **公式**：`Attention(Q,K,V) = Softmax(scale · Mask(Q·K^T, atten_mask) [+pse]) · V`（README L24-32）。
- **输入**：query/key/value（必选，FP16/BF16/INT8；FP8/FP4 见 V3）；pse_shift/atten_mask/actual_seq_lengths(_kv)/deq_scale1/quant_scale1/deq_scale2/quant_scale2/quant_offset2（可选）；V2/V3 扩展：blocktable/antiquant/sharedPrefix/queryRope/keyRope/dequantScaleQuery/learnableSink（def.cpp L26-75，kernel 入口 L25-37）。
- **输出**：attention_out（必选）；softmaxLse（推理，kernel 入口 L36）。
- **属性**：num_heads(必需)、num_key_value_heads(默认0)、scale_value(默认1.0)、pre_tokens(默认214748647)、next_tokens(默认0)、input_layout(默认BSH)、sparse_mode(默认0)、inner_precise(默认1)（def.cpp L80-103）。
- **约束**：B≤65536（不对齐时128）、N≤256、S≤20971520、D≤512；layout BSH/SBH/BNSD/BSND/TND/NTD/PA；TND 限定 N∈{8,16,32,64,128}、Q_D=K_D=192、V_D=128/192、BF16（README L94-166）。
- **变体**：V1/V2/V3；MLA（D≠Dv 或 D>256，D=576 nope+rope）；PagedAttention（blocktable）；共享前缀；解耦 RoPE。
- **空输入**：query 空 -> 返回；query 非空但 KV 空 -> 输出全零（README L92）。

## 4. Stage / pipeline table

| Stage | Entry（arch35 service/） | Unit | Op |
|---|---|---|---|
| Cube1 (Bmm1) | `prompt_flash_attention_mm1.h` | AIC | Q·K^T -> attention_score（FP32 累加） |
| Vector1 (Vec1) | `prompt_flash_attention_vector1.h` | AIV | pse -> ×scale -> atten_mask select -> FlashSoftmax(在线 max/sum) |
| Cube2 (Bmm2) | `prompt_flash_attention_mm2.h` | AIC | P·V matmul |
| Vector2 (Vec2) | `prompt_flash_attention_vector2.h` | AIV | rescale(÷softmax_sum) -> 可选 quant -> 写回 |

在线 FlashSoftmax：`exp[i]=e^(max[i-1]-max[i])`，`PV[i]=PV[i-1]·exp[i]+PV_new[i]`，`out=PV_final/softmax_sum`（设计介绍 L13-22）。

## 5. Tiling parameters

CANN 原始 tiling 详表见 `artifacts/prompt-flash-attention/tiling_dispatch_tables.md`。要点：
- Cube 基本块 128×128，Vector 8×1024，nRatio=8（1:16 CV 配比）（设计介绍 L59）。
- S1 factor 默认 128（CVDIFF_SOUTER_FACTOR_DEFAULT）、S2 factor 默认 1024（CVDIFF_SINNER_FACTOR_DEFAULT）（tiling.cpp L103-106）。
- mm1Res UB = 16384B = 128×128（tiling.cpp L102）；BLOCK_SIZE_BASE=128/MAX=1024（L93-94）。
- arch32 S1 负载候选 COF={256,384,512,640,768,896,960,1024}（tiling_const.h L63）。
- 值 source-reported，最优值 `needs_verification`。

## 6. Core partition strategy

- 逻辑轴 B/N2/G/S1/S2，先核内再核间（设计介绍 L49）。
- splitCoreMode：`actualSeqLength≥3072 && b·n·sOuter≥coreNum` -> SPLIT_ONEN_CUBE，否则 SPLIT_NBS_CUBE（tiling.cpp L926-932）。
- 负载均衡：`singleCoreSize = ceil(totalSize/coreNum)`（tiling.cpp L713）。
- 三原则：多核用满 / 负载均匀 / CV 算力匹配（设计介绍 L43-47）。
- 标准路径 S2 不切到多核（FlashSoftmax 需连续 S2 在线更新）；PA/FlashDecoding 场景 `needs_verification`。

## 7. Kernel templates & dispatch

- **arch 分发**：ascend910b/910_93 -> arch32（opFile `prompt_flash_attention`）；ascend950 -> arch35（opFile `prompt_flash_attention_apt`）；ascend310p -> 310P 分支（def.cpp L114-127，tiling.cpp L429/433/1469）。
- **arch32 tilingkey**：~130 个十进制组装宏（dtype×layout×precision×API×feature，tilingkey.h；规则见设计介绍 L61-92）。CANN 编译期模板注册机制，**不迁移**。
- **arch35 模板参数化**：结构化枚举（Layout/S1/S2/D/DV/ImplMode/PseMode/QuantMode/Mask/MatMulType/SplitCoreMode）+ 27 个 ConfigValue（template_tiling_key_enum.h）。`if constexpr` 分发。
- **MLA 触发**：`D != Dv` 或 `D > 256`（tiling.cpp L2848）。
- **空 KV**：tilingkey=20（tiling.cpp L73）。
- 可迁移：arch 选择逻辑、MLA/PA/quant 触发条件、基本块概念。不迁移：十进制 tilingkey、二进制注册、平台查询。

## 8. Pipeline analysis

详表见 `artifacts/prompt-flash-attention/pipeline-diagram.md`。要点：
- 四阶段 CV 流水 Bmm1->Vec1->Bmm2->Vec2，与训练 FA 同骨架（设计介绍 L9-22）。
- 在线 FlashSoftmax（可迁移数学语义）：running max/sum + exp rescale + PV 累加。
- V 侧 double buffer（ping-pong）；Cube 双发机制（设计介绍 L99-138，CANN 专有，抽象为 tile 复用/fused load-store）。
- UB：mm1Res/softmaxMax/Sum/Exp/mm2Res/pse/attenMask/apiTmp（FP32，最大 32KB）。
- Workspace：`sysWorkspace(16MB) + coreNum×(lse/max/sum + mm1Res×2 + mm2Res×2)`（tiling.cpp L465-490，×2=double buffer）。
- PFA 相对训练 FA 差异：推理语义、softmaxLse 输出、KV-cache flashupdate（`needs_verification`）、PA、量化、MLA、共享前缀、解耦 RoPE。

## 9. CANN -> Triton-Ascend migration classification

完整四分类表（transferable / rewrite / CANN-specific / needs-verification，每条带 CANN 来源 path Lxx）见 `artifacts/prompt-flash-attention/migration-table.md`。概要：
- **transferable**：注意力公式、在线 FlashSoftmax rescale、mask select->-inf、pse add/mul、scale、FP32 累加、空输入处理、变长 actualSeqLengths。
- **rewrite**：S1/S2 基本块->`tl.constexpr`、core id/splitCoreMode->`program_id`/grid、MLA D 拼接、tilingkey 分支->`if constexpr`、PA blocktable->间接 load、量化->cast+scale 融合、arch32/35 选择。
- **CANN-specific（不迁移）**：MTE 手动搬运、BufferID/queue/event/CV 手动同步、ND↔NZ 格式转换、Cube 双发/nRatio 配比、TBuf 放置/平台查询、十进制 tilingkey 注册。
- **needs_verification**：tiling 参数最优值、UB/L1 字节预算、FP8/FP4 精度边界、softmaxLse 精确公式、KV-cache flashupdate 写回时机、长序列超时阈值。

## 10. Triton-Ascend skeleton

见 `artifacts/prompt-flash-attention/prompt-flash-attention-triton-skeleton.py`。
`reproducibility: pseudocode`。表达：kernel 签名、`tl.constexpr` meta（BLOCK_M=128/BLOCK_N=1024，值 source-reported）、`tl.program_id` 映射、变长边界 mask、QK·FP32 dot、在线 softmax（running max/sum + exp rescale + PV 累加）、atten_mask select、可选 pse/PA/MLA/quant 分支、rescale 写回、softmaxLse 输出。未编译、未验证、未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `kernelwiki/sources/ops-transformer/prompt-flash-attention.md` | source | `ops-transformer-prompt-flash-attention` | - | - | - |
| `kernelwiki/wiki/kernels/prompt-flash-attention.md` | kernel | `kernel-prompt-flash-attention` | `[ops-transformer-prompt-flash-attention]` | inferred | pseudocode |
| `kernelwiki/artifacts/prompt-flash-attention/PROVENANCE.yaml` | provenance | - | - | - | - |
| `…/source_file_inventory.md` | approach-notes | - | - | inferred | - |
| `…/tiling_dispatch_tables.md` | approach-notes | - | - | inferred | - |
| `…/pipeline-diagram.md` | approach-notes | - | - | inferred | - |
| `…/migration-table.md` | approach-notes | - | - | inferred | - |
| `…/prompt-flash-attention-triton-skeleton.py` | derived-source | - | - | inferred | pseudocode |

kernel 页 `related` 引用既有 `kernel-flash-attention-score`、`technique-flash-attention-tiling`、`technique-flash-attention-pipeline`、`pattern-flash-attention-tiling-key-dispatch`。

## 12. Validation & retrieval

```
$ python kernelwiki/scripts/validate.py
OK: KernelWiki validation passed

$ python kernelwiki/scripts/generate-indices.py
Generated queries/{by-problem,by-technique,by-hardware-feature,by-repo,by-kernel-type}.md

$ python kernelwiki/scripts/query.py prompt_flash_attention --compact   # 3 hits：新源页 + 新 kernel 页 + FA kernel 页 ✓
$ python kernelwiki/scripts/query.py attention tiling --compact         # 10 hits，含 kernel-prompt-flash-attention ✓
$ python kernelwiki/scripts/query.py attention mask --compact           # 10 hits，含 kernel-prompt-flash-attention ✓
$ python kernelwiki/scripts/query.py attention triton ascend --compact  # 10 hits（Triton-Ascend 文档）
```

PROVENANCE 5 文件 sha256 已由 `scaffold.py update-provenance` 冻结；validate 通过（无缺失 PROVENANCE、无 sha256 不匹配、无未解析 source、无未知词表值）。

## 13. Unverified / needs-hardware items

| 项 | 原因 |
|---|---|
| tiling 参数最优值（S1.i=128 等） | 依赖运行时 shape/平台，源码仅给默认/候选 |
| UB/L1/L0C 字节预算/占用 | 需硬件实测 |
| FP8/FP4 精度边界 | 需运行时验证 |
| softmaxLse 精确公式 | 以 `vf_softmaxflashv3.h` 源码为准，未逐行确认 |
| KV-cache flashupdate 写回时机 | prefill 是否同时写 cache 待确认 |
| 长序列超时阈值 | 依赖 B·S·N·D 乘积，无固定值 |
| TND/PA offset 算术 | 需张量实例确认 |
| Ascend310P 精确芯片型号 | tags.yaml 有 310P1/3/5/7，未确认具体型号，故 architectures 未含 |
| V2/V3 相对前版的完整 diff | 仅 skim 文档，未逐条核对 |

## 14. Modified / created files

**新建：**
- `kernelwiki/sources/ops-transformer/prompt-flash-attention.md` - 来源摘要页
- `kernelwiki/wiki/kernels/prompt-flash-attention.md` - kernel 知识页
- `kernelwiki/artifacts/prompt-flash-attention/PROVENANCE.yaml` - 来源溯源（scaffold init 生成 + update-provenance 冻结）
- `kernelwiki/artifacts/prompt-flash-attention/source_file_inventory.md` - 源文件清单
- `kernelwiki/artifacts/prompt-flash-attention/tiling_dispatch_tables.md` - tiling/dispatch 分析表
- `kernelwiki/artifacts/prompt-flash-attention/pipeline-diagram.md` - 流水分析表
- `kernelwiki/artifacts/prompt-flash-attention/migration-table.md` - 迁移四分类表
- `kernelwiki/artifacts/prompt-flash-attention/prompt-flash-attention-triton-skeleton.py` - Triton skeleton（pseudocode）
- `reports/prompt-flash-attention-ingest-report.md` - 本报告

**修改：**
- `kernelwiki/queries/by-{problem,technique,hardware-feature,kernel-type,by-repo}.md` - `generate-indices.py` 重新生成

## Quality checklist

- [x] 每个 `sources:` 解析到真实 source 页 id；wiki 页引用 source。
- [x] `architectures/hardware_features/techniques/kernel_types/symptoms` 均为 `tags.yaml` 词表值。
- [x] `validate.py` 通过。
- [x] `performance_claims: []`（无 baseline/env/shapes/repeats/result）。
- [x] `confidence: inferred`、`reproducibility: pseudocode`（非 verified/benchmarked）。
- [x] 全文无"已运行/已验证/benchmarked"声明。
- [x] CANN 专有机制（MTE/BufferID/queue/event/手动同步/ND↔NZ）未作为 Triton 实现指南，已抽象。
- [x] 报告列出全部新建/修改文件。
