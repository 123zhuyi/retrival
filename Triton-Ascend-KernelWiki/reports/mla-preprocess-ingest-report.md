# mla_preprocess CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `kernelwiki/` | source-confirmed |
| OPERATOR_NAME | `mla_preprocess` | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred（源自 `ops-transformer/attention/`） |
| VARIANTS | inference（MLA 前处理融合：norm+matmul+rope+cache） | inferred |
| commit | `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory

算子目录 `attention/mla_preprocess/`，~30 文件，无 arch 子目录（单架构 A2/A3）。核心：`README.md`(371L)、`docs/aclnnMlaPreprocess.md`(1272L)、`op_host/mla_preprocess_def.cpp`(122L)、`op_host/mla_preprocess_tiling.cpp`(708L)、`op_host/mla_preprocess_tiling{,data}.h`(68/103L)、`op_kernel/mla_preprocess.cpp`(1206L)、`op_kernel/mla_preprocess_{fp16,bf16,no_quant}.h`(2656/3093/2365L)、9 个 `*_iterator.h` buffer 搬运迭代器、`mla_{mma,simd,kernel_utils,common,mem,layout,hardware}.h` 公共组件。

完整清单（含 SHA-256）见 `artifacts/mla-preprocess/source_file_inventory.md`。

## 3. Operator semantics

- **功能**：推理场景 MLA（Multi-Head Latent Attention）前处理融合算子。x -> RmsNormQuant -> ×W^DQKV(降维) -> 拆 Q/K；Q: RmsNormQuant -> ×W^UQ -> 拆 nope/rope -> nope ×W^UK -> q^N，rope RoPE -> q^R；K: RmsNorm -> cache k^N，RoPE -> cache k^R（README L12-19）。
- **公式**：`RmsNormQuant(x)=(RmsNorm(x)+bias)·deqScale`；`q^N=RmsNormQuant(x)·W^DQKV·W^UK`（MLA 权重吸收 2-matmul 形式）；`q^R=ROPE(x^Q)`；`k^N=Cache(RmsNorm(RmsNormQuant(x)))`；`k^R=Cache(ROPE(RmsNormQuant(x)))`（README L20-54）。
- **输入**：input/gamma_0/beta_0/quant_scale_0/quant_offset_0/wdqkv/descale_0/bias_0/gamma_1/beta_1/quant_scale_1/quant_offset_1/wuq/descale_1/bias_1/gamma_2/cos/sin/wuk/kv_cache/kv_cache_rope/slot_mapping/ctkv_scale/q_nope_scale（多为 REQUIRED，FP16/BF16/INT8，ND/NZ 混合）（def.cpp L73-96）。
- **输出**：q_out/kv_cache_out/q_rope_out/kr_cache_out（def.cpp L97-100）。
- **属性**：wdq_dim(128)/q_rope_dim(64)/k_rope_dim(64)/epsilon(1e-5)/q_rotary_coeff(2)/k_rotary_coeff(2)/transepose_wdq/wuq/wuk(true)/cache_mode(0-3)/quant_mode(0-3)/do_rms_norm(true)/wdkv_split_count(1)（多项预留只支持单值）（def.cpp L101-113）。
- **约束**：tokenNum 0-256；hiddenSize 2048-10240（256 倍数）；headNum {16,32,64,128}；blockNum 192；blockSize 128；BF16 权重 -> input BF16 且 hiddenSize 仅 6144（README L363-371）。
- **与 MLA attention 关系**：本算子是 [[kernel-sparse-flash-attention]]/[[kernel-prompt-flash-attention]] MLA-absorb（attention_mode=2）的上游预处理，输出 q^N(512)+q^R(64) 与 k^N/k^R cache；本身不做 softmax/QK/PV。

## 4. Stage / pipeline table

| Stage | Unit | Op | tiling 方法 |
|---|---|---|---|
| Cube (ProcessCube) | AIC | 3 matmul（mm1 W^DQKV / mm2 W^UQ / mm3 W^UK）+ RoPEConcat | RopeConcatTiling |
| Vector (ProcessVector) | AIV | 2× RmsNormQuant + RmsNorm₃ + EinSumQuant + Cache 写回 | RmsNormQuantTiling / EinSumQuantTiling |

kernel 入口先 `ProcessCube()` 再 `ProcessVector()`（mla_preprocess.cpp L52-53）。buffer 搬运由 9 个 `*_iterator.h`（GM/L1/L0C/UB/BT/FB 间）封装（CANN 专有）。

## 5. Tiling parameters

详见 `artifacts/mla-preprocess/tiling_dispatch_tables.md`。要点：
- dtype dispatch：FP16 -> `_fp16.h`；BF16 -> `_bf16.h` + `_no_quant.h`（L17-22）。
- tiling key = cacheMode × 3 输出格式(ND/NZ)（L46+，步长 4，8 组合/cacheMode）。
- 3 matmul tiling mm1/mm2/mm3（MlaPpMatmulTilingData：m/k/n + m0/k0/n0 基本块 + loop + swizzle + ping-pong，tilingdata L25-43）。
- 核间切分：tokenNum 沿 numCore，perTaskNum/resTaskNum（L47-50）。
- workspace 3 等大 section（s1/s2/s3，L41-43）。
- 值 source-reported，最优值 `needs_verification`。

## 6. Core partition strategy

tokenNum 沿 numCore 切分，每核 perTaskNum、尾核 resTaskNum；前核/尾核循环次数 preCoreLoopTime/lastCoreLoopTime 负载均衡（tilingdata L75-79）。Cube/Vector CV 分离（Cube matmul，Vector norm/quant）。

## 7. Kernel templates & dispatch

- 3 个 dtype/quant 模板：`mla_preprocess_fp16.h` / `_bf16.h` / `_no_quant.h`。
- tiling key 编码 cacheMode(KVCACHE=0-28 / KROPE_CTKV=32-60) × 3 输出格式(ND/NZ)，模板参数 `<CacheMode, F1, F2, F3>`。
- 可迁移：dtype/cacheMode/format 分支逻辑、3 matmul 结构。不迁移：tiling key 数值、buffer iterator、CV 分离机制。

## 8. Pipeline analysis

详见 `artifacts/mla-preprocess/pipeline-diagram.md`。要点：
- 6 通路数据流：x -> RmsNormQuant₁ -> mm1 -> 拆 Q/K；Q 路 RmsNormQuant₂ -> mm2 -> 拆 nope(rope)/rope -> mm3/RoPE -> q^N/q^R；K 路 RmsNorm₃/RoPE -> cache k^N/k^R。
- CV 分离：Cube 3 matmul + RoPEConcat，Vector RmsNormQuant + EinSumQuant + Cache scatter。
- workspace 3 section；Cache 按 slot_mapping scatter 写回，cacheMode 决定格式/量化。
- buffer iterator（GM/L1/L0C/UB）+ ND↔NZ + swizzle/ping-pong 为 CANN 专有，抽象为 fused load/store + 单 program。

## 9. CANN -> Triton-Ascend migration classification

完整四分类表见 `artifacts/mla-preprocess/migration-table.md`。概要：
- **transferable**：RmsNorm/RmsNormQuant 公式、3 matmul、RoPE、MLA 权重吸收等价、Cache scatter、FP32 累加、尾核边界、quantMode/cacheMode 语义。
- **rewrite**：tokenNum 核间切分->program_id、mm 基本块->constexpr、tiling key 分支->if constexpr、dtype dispatch。
- **CANN-specific**：ND↔NZ 输出格式、`*_iterator.h` buffer 搬运、ProcessCube/Vector CV 分离、swizzle/ping-pong/shuffleK、`__DAV_C220_CUBE__` 平台 guard。
- **needs_verification**：tiling 最优值、workspace 预算、MLA 吸收实现、quantMode 1/2/3（未实现）、cacheMode=2 int8 量化边界。

## 10. Triton-Ascend skeleton

见 `artifacts/mla-preprocess/mla-preprocess-triton-skeleton.py`。`reproducibility: pseudocode`。表达：RmsNormQuant + 3 个 `tl.dot` + RoPE + slot_mapping scatter store + cacheMode/quantMode 分支。未编译/未验证/未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `sources/ops-transformer/mla-preprocess.md` | source | `ops-transformer-mla-preprocess` | - | - | - |
| `wiki/kernels/mla-preprocess.md` | kernel | `kernel-mla-preprocess` | `[ops-transformer-mla-preprocess]` | inferred | pseudocode |
| `artifacts/mla-preprocess/PROVENANCE.yaml` | provenance | - | - | - | - |
| `…/source_file_inventory.md` | approach-notes | - | - | inferred | - |
| `…/tiling_dispatch_tables.md` | approach-notes | - | - | inferred | - |
| `…/pipeline-diagram.md` | approach-notes | - | - | inferred | - |
| `…/migration-table.md` | approach-notes | - | - | inferred | - |
| `…/mla-preprocess-triton-skeleton.py` | derived-source | - | - | inferred | pseudocode |

kernel 页 `related` 引用 `kernel-sparse-flash-attention`、`kernel-prompt-flash-attention`、`kernel-flash-attention-score`（MLA 上下游）。

## 12. Validation & retrieval

```
$ python kernelwiki/scripts/validate.py
OK: KernelWiki validation passed

$ python kernelwiki/scripts/generate-indices.py   -> OK

$ python kernelwiki/scripts/query.py mla_preprocess --compact
# 2 hits：source 页 + kernel-mla-preprocess(新) ✓

$ python kernelwiki/scripts/query.py attention mla --compact
# 10 hits：kernel-mla-preprocess + PFA + SFA + incre-FA + block-sparse-FA + FA + sparse 技巧页 ✓

$ python kernelwiki/scripts/query.py rmsnorm quant --compact
# 10 hits：含 kernel-mla-preprocess（norm+matmul 融合特性）✓
```

PROVENANCE 5 文件 sha256 已冻结；validate 通过。

## 13. Unverified / needs-hardware items

| 项 | 原因 |
|---|---|
| tiling 参数最优值（m0/k0/n0 等） | 依赖运行时 shape/平台 |
| workspace 字节预算（maxWorkspaceSize） | 需硬件实测 |
| MLA 权重吸收实现 | 数学 2-matmul vs kernel 3-matmul，是否运行时吸收待确认 |
| quantMode 1/2/3 | README L315 标注未实现，仅 0 实现 |
| cacheMode=2 int8 per_head 对称量化边界 | 需运行时验证 |
| Cube/Vector 内部子阶段排布（double buffer 等） | 在 `_fp16.h`/`_bf16.h` 内，未逐行确认 |
| 预留参数（wdq_dim/transpose_*/split_count） | 只支持单值，固定 |

## 14. Modified / created files

**新建：**
- `kernelwiki/sources/ops-transformer/mla-preprocess.md` - 来源摘要页
- `kernelwiki/wiki/kernels/mla-preprocess.md` - kernel 知识页
- `kernelwiki/artifacts/mla-preprocess/{PROVENANCE.yaml, source_file_inventory.md, tiling_dispatch_tables.md, pipeline-diagram.md, migration-table.md, mla-preprocess-triton-skeleton.py}` - artifact bundle
- `reports/mla-preprocess-ingest-report.md` - 本报告

**修改：**
- `kernelwiki/queries/by-*.md` - `generate-indices.py` 重新生成

## Quality checklist

- [x] `sources:` 解析到真实 source 页；kernel 页引用 source。
- [x] `architectures/hardware_features/techniques/kernel_types/symptoms` 均为 `tags.yaml` 词表值。
- [x] `validate.py` 通过。
- [x] `performance_claims: []`。
- [x] `confidence: inferred`、`reproducibility: pseudocode`。
- [x] 无"已运行/已验证/benchmarked"声明。
- [x] CANN 专有机制（buffer iterator/ND↔NZ/CV 分离/swizzle）未作为 Triton 指南，已抽象。
- [x] 报告列出全部新建/修改文件。
