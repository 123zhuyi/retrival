# mla_prolog CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `kernelwiki/` | source-confirmed |
| OPERATOR_NAME | `mla_prolog` | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred |
| VARIANTS | inference（MLA prolog V1，4 matmul+RmsNorm+RoPE+cache） | inferred |
| commit | `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory

算子目录 `attention/mla_prolog/`，~40 文件，arch32+arch35 双架构。核心：`README.md`(361L)、`docs/MlaProlog算子设计介绍.md`(454L，含完整 Process 伪代码)、`op_host/mla_prolog_def.cpp`(271L)、`op_host/mla_prolog_tiling.cpp`(841L)、`op_kernel/mla_prolog.cpp`(100L)、`op_kernel/kernel_mla_prolog_split_n.h`(2241L)、`op_kernel/mla_prolog_template_tiling_key.h`(411L)、`op_kernel/service_{matmul,rms_norm,rope,scatter_cache,dequant,rotary_position_embedding}.h`、`op_kernel/arch35/vf/*.h`(6 个 Vector 服务)。完整清单（含 SHA-256）见 `artifacts/mla-prolog/source_file_inventory.md`。

## 3. Operator semantics

- **功能**：推理 MLA 前处理融合 V1。四路：x→×W^DQ(RmsNorm)→[×W^UQ·W^UK→q^N]+[×W^QR→RoPE→q^R]+[×W^DKV→RmsNorm→cache k^C]+[×W^KR→RoPE→cache k^R]（README L12-60）。
- **公式**：`q^N=RmsNorm(x·W^DQ)·W^UQ·W^UK`、`q^R=ROPE(RmsNorm(x·W^DQ)·W^QR)`、`k^R=ROPE(x·W^KR)`、`c^KV=RmsNorm(x·W^DKV)`（设计介绍 L149-166）。
- **输入**：token_x/weight_dq/weight_uq_qr/weight_uk/weight_dkv_kr/rmsnorm_gamma_cq/ckv/rope_sin/cos/cache_index/kv_cache/kr_cache（REQUIRED）+ dequant/quant/smooth_scales（OPTIONAL，预留参数未使用）（def.cpp L23-117）。
- **输出**：query/query_rope/kv_cache/kr_cache（def.cpp L118-133）。
- **属性**：rmsnorm_epsilon_cq/ckv(1e-5)、cache_mode("BNSD")（def.cpp L134-136）。
- **约束**：3 场景（非量化/部分量化KV非量化/部分量化KV量化）；B/S/T/Skv=0→空 tensor；A2/A3+950(FP8)（README L91-128）。
- **与 mla_preprocess 差异**：V1 权重更细粒度（W^DQ/W^UQ/W^QR/W^DKV/W^KR 独立），支持 950 FP8，CV 同步 6 event；V2+ 权重合并为 W^DQKV/W^UQ/W^UK，仅 A2/A3。

## 4. Stage / pipeline table

7 阶段 CV 流水（6 sync event，设计介绍 L83-146）：

| # | 阶段 | 单元 | 操作 | 同步 |
|---|---|---|---|---|
| 1 | MatmulCq | AIC | x·W^DQ | →SYNC_MMCQ_NORMSCQ |
| 2 | MatmulCkvKr | AIC | x·[W^DKV\|W^KR] | →SYNC_MMCKVKR_NORMROPE |
| 3 | RmsNormCq | AIV | γ·cQ/RMS(cQ) | ←1, →全核 |
| 4 | RmsNormRopeScatterCkvKr | AIV | RmsNorm(cKV)+RoPE(kR)+Cache | ←2 |
| 5 | MatmulQcQr | AIC | cQ_norm·[W^UQ\|W^QR] | ←3, →全核 |
| 6 | RopeQr | AIV | RoPE(qR) | ←5 |
| 7 | MatmulQn | AIC | qC·W^UK | ←全核(5) |

## 5. Tiling parameters

详见 `artifacts/mla-prolog/tiling_dispatch_tables.md`。tiling key 二进制编码（14 bits：CACHE_MODE+SCENARIO+QUANT_MODE+DEQUANT/COMPUTE+EMPTY）。8 模板参数。3 场景。4 matmul 切分（BS 合轴 M，N 按 Cube 核数）。MIX_AIC_1_2。

## 6. Core partition strategy

BS 合轴做 M 维，N 轴按 Cube 核数切分（MatmulCq 按 Hcq / MatmulCkvKr 按 Hckv+Dr / MatmulQcQr 按 N*(D+Dr)）。MatmulQn 依赖全核 MatmulQcQr 完成（全核同步 SYNC_ALL_CUBE）。

## 7. Kernel templates & dispatch

3 量化场景 × 2 架构（A2/A3 + 950）→ 模板组合。tiling key 二进制定位。可迁移：场景分支逻辑、4 matmul 结构、切分策略。不迁移：tiling key 二进制位、FRACTAL_NZ 格式、6 CV sync event。

## 8. Pipeline analysis

详见 `artifacts/mla-prolog/pipeline-diagram.md`。核心：6 CV sync event（CANN 专有）— 4 matmul 在 Cube、2 RmsNorm+2 RoPE+cache 在 Vector，通过全核同步和核间 event 协调。W^UQ+W^QR 与 W^DKV+W^KR 分别合并 matmul 后 split。RoPE 优化公式(15)：奇偶分拆 cos/sin 乘。

## 9. Migration classification

完整四分类表见 `artifacts/mla-prolog/migration-table.md`。概要：
- **transferable**：4 matmul 公式、RmsNorm、RoPE、权重合并 matmul 后 split、Cache scatter、FP32 累加、3 量化场景语义、空 tensor 分支。
- **rewrite**：tiling key 二进制位→if constexpr、8 模板参数、BS 合轴 M→program_id、matmul 基本块→constexpr、950 FP8 分支。
- **CANN-specific**：6 CV sync event、ND↔NZ cache 格式、MIX_AIC_1_2、FRACTAL_NZ 权重格式。
- **needs_verification**：tiling 最优值、FP8 精度边界、MLA 权重吸收实现、预留参数（未使用）。

## 10. Triton-Ascend skeleton

见 `artifacts/mla-prolog/mla-prolog-triton-skeleton.py`。`reproducibility: pseudocode`。表达：4 matmul（含 W^UQ+W^QR/W^DKV+W^KR 合并→split）+ 2 RmsNorm + 2 RoPE + 2 Cache scatter + 3 量化场景分支。未编译/未验证。

## 11. KB pages written

| Path | type | id |
|---|---|---|
| `sources/ops-transformer/mla-prolog.md` | source | `ops-transformer-mla-prolog` |
| `wiki/kernels/mla-prolog.md` | kernel | `kernel-mla-prolog` |
| `artifacts/mla-prolog/{PROVENANCE.yaml,source_file_inventory.md,tiling_dispatch_tables.md,pipeline-diagram.md,migration-table.md,mla-prolog-triton-skeleton.py}` | bundle | - |

kernel 页 `related` 引用 `kernel-mla-preprocess`（V1 vs V2+ 对比）、`kernel-sparse-flash-attention`、`kernel-prompt-flash-attention`、`kernel-flash-attention-score`。

## 12. Validation & retrieval

```
validate.py        -> OK
generate-indices   -> OK
query mla_prolog          -> 2 hits（source + kernel）✓
query mla preprocess      -> 10 hits（mla_prolog+mla_preprocess 配对 + MLA 系列）✓
query prolog              -> 2 hits ✓
```

PROVENANCE 5 文件 sha256 已冻结。

## 13. Unverified / needs-hardware items

tiling 最优值、FP8 精度边界、MLA 权重吸收、预留参数（未使用）。

## 14. Modified / created files

**新建**：`sources/ops-transformer/mla-prolog.md`、`wiki/kernels/mla-prolog.md`、`artifacts/mla-prolog/`(6 文件)、`reports/mla-prolog-ingest-report.md`。

**修改**：`kernelwiki/queries/by-*.md`（reindex）。

## Quality checklist

- [x] `sources:` 解析到真实 source 页
- [x] 词表值合规；validate 通过
- [x] `performance_claims: []`；`confidence: inferred`；`reproducibility: pseudocode`
- [x] 无"已运行/已验证/benchmarked"声明
- [x] CANN 专有机制（6 CV sync event/ND↔NZ/FRACTAL_NZ/MIX_AIC_1_2）未作为 Triton 指南