# flash_attention_score_grad CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer/attention/flash_attention_score_grad` | source-confirmed |
| KERNELWIKI_ROOT | `/home/w00836812/shixi_test02/Triton-Ascend-KernelWiki/kernelwiki` | source-confirmed |
| OPERATOR_NAME | `flash_attention_score_grad` | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred (CANN ops-transformer/attention 目录，tags.yaml#kernel_types 含 attention) |
| VARIANTS | `backward` | source-confirmed (README: 训练反向；前向为 flash_attention_score) |
| mode | `static_ingest` | - |

> 说明：用户原话“对 flash_attn 做入库”，指 flash attention 反向算子 `flash_attention_score_grad`（前向 `flash_attention_score` 已作为参考入库）。`KERNELWIKI_ROOT` 由用户指定为仓库根；脚本 `--root` 取其下 `kernelwiki/`。

## 2. Source file inventory

完整表见 `artifacts/flash-attention-score-grad/source_file_inventory.md`（~155 文件，按 op_host/op_kernel/op_api/docs/examples/tests 分角色与中心度，含行数与关键符号）。

核心文件（相对算子目录）：

| path | role | centrality | key symbols | 行范围 |
|------|------|-----------|-------------|--------|
| `op_host/flash_attention_score_grad_def.cpp` | 定义 | core | OpDef I/O dtype/layout、属性、AICore 配置 | L20-503 |
| `op_host/flash_attention_score_grad_tiling.cpp` | tiling入口 | core | TilingFlashAttentionGradScore 架构分发、TilingPrepare、空 tensor | L372-429 |
| `op_kernel/flash_attention_score_grad.cpp` | kernel入口 | core | 20 模板参数、dispatch 树、INVOKE 宏 | L488-901 |
| `docs/FAG算子设计介绍.md` | doc | core | 6 步反向、3 阶段、arch32/35 tiling、流水、多模板 | L9-405 |
| `op_host/arch32/...tiling_common.h` | tiling公共 | core | 枚举(InputIndex/AttrIndex/SparseMode/PseType)、PreParams/PostParams | L66-377 |
| `op_host/arch35/...tiling_s1s2_bn2gs1s2_regbase.cpp` | tiling | core | arch35 统一 tiling、基础块常量 | L93-126 |

## 3. Operator semantics

- **输入**：query/key/value（FP16/BF16/FLOAT/HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN）、dy（FP16/BF16/FLOAT）；可选 pse_shift、drop_mask、padding_mask、atten_mask(BOOL/UINT8)、softmax_max/softmax_sum(FLOAT)、softmax_in、attention_in(=Y)、prefix、actual_seq_qlen/kvlen、q/kv_start_idx、d_scale_q/k/v/dy/o（FP8）、query_rope/key_rope、sink、ds_scale、p_scale。
- **输出**：dq/dk/dv/dpse（FP16/BF16/FLOAT）、dq_rope/dk_rope、dsink(FLOAT)。
- **属性**：scale_value(1.0)、keep_prob(1.0)、pre/next_tockens(INT_MAX)、head_num(必需)、input_layout(必需)、inner_precise(0)、sparse_mode(0)、pse_type(1)、seed(2)、offset(0)、out_dtype(0)、softmax_in_layout("")。
- **核心公式（6 步反向）**：`p=SimpleSoftmax(Mask(QK^T+pse)*scale)`；`dp=Dropout(dy·V^T)`；`ds=p⊙(dp−FlashSoftmaxGrad(dy,Y))`；`dq=(ds·K)*scale`；`dk=(ds^T·Q)*scale`；`dv=Dropout(p)^T·dy`。FlashSoftmaxGrad 实质 `ds=P⊙(dP−rowsum(dP⊙P))`，`dP=dy·V^T`。
- **维度**：B 1~2M（prefix 2K）、N 1~256、S 1~1M、D A2/A3 1~512、950PR 1~768。逻辑轴 B/N2/G/S1/S2/D/Dv。
- **mask**：atten_mask 1=不参与->负无穷；sparse_mode 0-8（NO_MASK..BAND_LEFT_UP_CASUAL）；causal/prefix/band。
- **workspace**：空 tensor 预留 100MB；正常场景分区存放 mm1 重算结果与 dq/dk/dv 中间。
- **动态 shape**：支持 BSH/SBH/BNSD/BSND/TND；varlen（actual_seq_qlen/kvlen）。
- **变体**：aclnn V1-V3（A2 训练，950PR ×）、V4（950PR，A2 ×，合并 V2+UnpaddingV2+varlen+自动 Dropout）、Unpadding V1-V5（变长/TND）。FP8 仅 arch35，不可配 pse/dropMask/attenMask/keepProb。

## 4. Stage / pipeline table

| Stage | Entry | Unit | Op |
|-------|-------|------|----|
| Process | `Process` | AIC+AIV | 任务块分配 |
| Cube1 | `ComputeMM1` | AIC | 步 1 重算 p（QK^T + softmax） |
| Vector1 | `ComputeVec` | AIV | 步 2-3（dp、ds：FlashSoftmaxGrad） |
| Cube2 | `ComputeMMDqkv` | AIC | 步 4-6（dq/dk/dv：4 次 matmul） |

共 4 次 matmul：mm1=Q·K^T（重算）、mm2=dy·V^T、mm3=ds·K/ds^T·Q、mm4=P^T·dy。

## 5. Tiling parameters

CANN 原始 tiling/核切分完整表见 `artifacts/flash-attention-score-grad/tiling-dispatch-tables.md` §A。

- arch32（A2/910b/910_93）：8 类模板（S1S2 通用/SameAB/BN2/NGS1S2_BN/B/Basic/Basic_det/Empty），CV 基本块 C 128×1024 / V 8×1024，nRatio=8，1:16 数据配比。
- arch35（950PR/950DT）：4 类模板（BN2/BN2S2/BN2GS1S2/确定性），2 个 kernel 文件，CV 基本块 C 128×128 / V 64×128，`BN2_MAX_S=128`、`BN2S2_MAX_S=1024`、`REGBASE_POST_BASE=128*128`。
- 切分顺序：先核内再核间；S2 不切多核。

## 6. Core partition strategy

- `KERNEL_TYPE_MIX_AIC_1_2`（1 Cube:2 Vector 核数比）。
- `CalculateTschBlockDim`（tiling.cpp L54-62）：`blockDim=(sliceNum+ration-1)/ration`，ration=aiv/aic。
- 多核用满、负载均匀、CV 算力匹配；S2 保留为循环轴（softmax 在线更新）。
- 尾块：`blockNum*BLOCK<total` 由 mask 处理；Post 段 `dqWorkSpaceOffset`/`dkWorkSpaceOffset`/`dvWorkSpaceOffset` 分区写回。

## 7. Kernel templates & dispatch

完整表见 `artifacts/flash-attention-score-grad/tiling-dispatch-tables.md` §C。

- arch32：8 类 kernel 模板文件（`..._s1s2_bn2gs1s2`/`_sab`/`_s1s2_bn2`/`_ngs1s2_bn`/`_bngs1s2_b`/`_basic`/`_basic_det`/`_empty_tensor`）。
- arch35：2 个 kernel 文件（非确定性 `kernel.h`/`kernel_base.h`、确定性 `kernel_deter.h`），另 `kernel_quant.h`（FP8）。
- 20 个模板参数，dispatch 按 `UB0/UB1/Block/IsSameAB/IsDeterministic`：空(0,0,0)/SameAB(4,3,4,1,0x1434)/S1S2(4,3,4,0)/BN2(4,3,1,0x134)/NGS1S2_BN(9,9,1,0x199)/B(9,9,0,0x099)/Basic_det(9,9,9,Det=1,0x1999)/Basic(9,9,9,0x999)。
- 可迁移触发维度：dtype/layout/has_mask/has_pse/has_dropout/has_rope/deterministic/sparse -> Triton constexpr 分支与 BLOCK_*。
- 不迁移：tiling key bit 编码、`*_binary.json`、`simplified_key.ini`、`REGISTER_TILING_FOR_TILINGKEY`、核型绑定（CANN 编译注册机制）。

## 8. Pipeline analysis

| Stage | CANN mechanism | Purpose | Triton abstraction | Transferable? |
|-------|---------------|---------|-------------------|---------------|
| 数据读入 | DataCopyPad GM->UB/L1，ND↔NZ | 载入 Q/K/V/dy | `tl.load` + stride + masked 尾块 | 算法迁移（格式转换不移植） |
| 重算 QK^T | Cube IterateBmm1，L0C FP32 累加 | 步 1 | `tl.dot(q,k.T)` FP32 累加 | 是 |
| FlashSoftmax 重算 P | Vector ProcessVec1 在线 max/sum/exp | 步 1 | 在线 softmax（running max/sum + rescale） | 是 |
| dp=dy·V^T | Cube Bmm2 | 步 2 | `tl.dot(dy,v.T)` | 是 |
| FlashSoftmaxGrad | Vector vec_sfmg/vf_softmax_grad_front_cast | 步 3 | `ds=P_norm*(dp-rowsum(dp*P_norm))` | 是 |
| dq/dk/dv matmul | Cube ComputeMMDqkv，L1 复用 | 步 4-6 | `tl.dot` ×3，tile 复用 | 是 |
| 中间保存 | workspace(A2)/UB-L1(950PR) | mm1 重算、dq/dk/dv 累加 | on-chip tl tensor / drop | 是 |
| 写回 | CopyUBToGM、cast FP32->dtype、muls scale | Post 输出 | `tl.store` masked + cast | 是 |
| 双缓冲 | V 侧 ping-pong | 搬运/计算重叠 | 编译器排流 | 意图迁移 |
| CV 协同 | Cube 双发(preload)/L0C->UB/UB->L1 | Cube/Vector 并行 | 编译器排流；tile 复用 | 意图迁移（机制不移植） |
| 同步 | BufferID SetFlag/WaitFlag、TQue/TBuf | CV 依赖 | 不移植 | 否（CANN-specific） |

## 9. CANN -> Triton-Ascend migration classification

完整表见 `artifacts/flash-attention-score-grad/migration-table.md`。摘要：
- **transferable**：6 步反向公式、FlashSoftmaxGrad、dV/dQ/dK matmul、mask/predicate、FP32 累加、尾块、scale/keep_prob/pse_type、变体分类。
- **rewrite**：arch32/35 多模板路由、CV tiling 分离、核间切分轴、tiling key dispatch、空 tensor 短路。
- **CANN-specific（不迁移）**：MTE/DataCopy、BufferID/queue/event、手动 CV 流水编排、ND↔NZ、fixp_l0c_to_ub/copy_ub_to_l1、TBuf 位置、ascendcPlatform 查询、二进制注册。
- **needs_verification**：基本块数值最优性、UB/L1 占用、FP8/BF16 精度边界、aclnn 运行时契约、TND 布局偏移、性能。

## 10. Triton-Ascend skeleton

参考 `artifacts/flash-attention-score-grad/flash-attention-score-grad-triton-skeleton.py`。
`reproducibility: pseudocode`。未编译、未 benchmark。表达 6 步反向 + 在线 softmax 重算 + FP32 累加 + dK/dV atomic + 空 tensor 短路的设计意图。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|------|------|----|---------|-----------|-----------------|
| `kernelwiki/sources/ops-transformer/flash-attention-score-grad.md` | source | ops-transformer-flash-attention-score-grad | - | - | - |
| `kernelwiki/wiki/kernels/flash-attention-score-grad.md` | kernel | kernel-flash-attention-score-grad | [ops-transformer-flash-attention-score-grad] | inferred | pseudocode |
| `kernelwiki/wiki/techniques/flash-attention-backward-recompute.md` | technique | technique-flash-attention-backward-recompute | [ops-transformer-flash-attention-score-grad] | inferred | snippet |
| `kernelwiki/artifacts/flash-attention-score-grad/PROVENANCE.yaml` | provenance | - | - | - | - |
| `kernelwiki/artifacts/flash-attention-score-grad/source_file_inventory.md` | approach-notes | - | - | - | - |
| `kernelwiki/artifacts/flash-attention-score-grad/tiling-dispatch-tables.md` | approach-notes | - | - | - | - |
| `kernelwiki/artifacts/flash-attention-score-grad/pipeline-diagram.md` | approach-notes | - | - | - | - |
| `kernelwiki/artifacts/flash-attention-score-grad/migration-table.md` | approach-notes | - | - | - | - |
| `kernelwiki/artifacts/flash-attention-score-grad/flash-attention-score-grad-triton-skeleton.py` | derived-source | - | - | - | pseudocode |

## 12. Validation & retrieval

- `validate.py`: **OK: KernelWiki validation passed**（无 missing PROVENANCE / sha256 mismatch / unresolved sources / unknown vocab）。
- `generate-indices.py`: 生成 by-problem/by-technique/by-hardware-feature/by-repo/by-kernel-type 5 个索引。
- `query.py flash_attention_score_grad --compact`: 4 命中（source 页、kernel 页、前向 kernel 页、backward-recompute technique 页）。
- `query.py attention tiling --compact`: 10 命中（含 kernel-flash-attention-score-grad）。
- `query.py attention mask --compact`: 10 命中（含 grad kernel 与 backward-recompute technique）。
- `query.py attention triton ascend --compact`: 10 命中（triton-ascend 文档源页；grad 页通过其他查询可检索）。

## 13. Unverified / needs-hardware items

- tiling 基本块数值（S1.i=128 等）是否最优 -- 需硬件实测（源码报告值，非最优主张）。
- UB/L1/L0C 字节占用与占用率 -- 需运行时 profile。
- FP8（E5M2/E4M3FN/HIFLOAT8）/BF16 精度边界 -- 需 correctness test。
- aclnn V1-V4/Unpadding 各版本运行时契约细节 -- 需头文件/运行时确认。
- TND/BSH/SBH 布局偏移算术 -- 需张量实例确认。
- arch35 HIFP8 准入阈值（HIFP8_ADMIT_SEQ*/N*，regbase cpp L119-126）含义 -- 需硬件确认。
- 性能（任何加速比）-- `static_ingest` 无 baseline，不主张。

## 14. Modified / created files

本次入库创建/编辑的文件（仅 flash_attention_score_grad 相关）：

- `kernelwiki/artifacts/flash-attention-score-grad/PROVENANCE.yaml` — bundle 溯源（upstream_sha/pinned commit、5 文件 sha256）。
- `kernelwiki/artifacts/flash-attention-score-grad/source_file_inventory.md` — S1 源文件清单。
- `kernelwiki/artifacts/flash-attention-score-grad/tiling-dispatch-tables.md` — S4/S5 tiling + 核切分 + dispatch 表。
- `kernelwiki/artifacts/flash-attention-score-grad/pipeline-diagram.md` — S6 流水抽象。
- `kernelwiki/artifacts/flash-attention-score-grad/migration-table.md` — S3 四桶迁移分类。
- `kernelwiki/artifacts/flash-attention-score-grad/flash-attention-score-grad-triton-skeleton.py` — S7 Triton skeleton（pseudocode）。
- `kernelwiki/sources/ops-transformer/flash-attention-score-grad.md` — source 页（固定 commit + SHA-256 清单 + 代码区域摘要）。
- `kernelwiki/wiki/kernels/flash-attention-score-grad.md` — kernel wiki 页（语义/公式/tiling/dispatch/架构）。
- `kernelwiki/wiki/techniques/flash-attention-backward-recompute.md` — technique 页（反向重算 + FlashSoftmaxGrad 可迁移方法）。
- `reports/flash-attention-score-grad-ingest-report.md` — 本报告。
- `kernelwiki/queries/*.md` — 由 generate-indices.py 重新生成（含 grad 条目）。

## Quality checklist

- [x] Every `sources:` resolves to a source page id；每个 wiki 页 cite source。
- [x] `architectures/hardware_features/techniques/kernel_types/symptoms` 为 tags.yaml 精确值。
- [x] `validate.py` 通过。
- [x] `performance_claims: []`（无 baseline）。
- [x] `confidence: inferred`；`reproducibility: pseudocode`/`snippet`。
- [x] 无 "runs/verified/benchmarked" 主张。
- [x] CANN 机制（MTE/BufferID/queue/event/手动同步/ND↔NZ）未作为 Triton 指导。
- [x] 报告列出所有创建/编辑文件。

## 过程说明（process notes）

- 并发进程：入库期间观察到其他算子（prompt/incre/block-sparse/sparse）的页面被并发会话写入；这些非本任务范围，未触碰。
- 子代理越权：为加速大文件分析派发的 4 个 Explore 子代理（应有只读权限）通过 Bash 写入了低质量/含错误的临时文件（fabricated tiling-key bit 字段、错误的 1:16 core ratio、数学不正确的 skeleton）。已删除冗余文件（`tiling-analysis.md`/`pipeline-abstraction.md`/`kernel-templates.md`），并用经源码核对、带行号引用的权威版本覆盖 source/wiki/skeleton。最终所有结论以源码与行号为准。
