---
id: ops-transformer-mla-prolog-v3
title: CANN ops-transformer MlaPrologV3 source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/mla_prolog_v3/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MlaPrologV3 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/mla_prolog_v3/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。`fixture: true`。MLA Prolog V3 是 DeepSeek V2/V3 核心创新 MLA（Multi-head Latent Attention）的前处理融合算子。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `attention/mla_prolog_v3/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期: 2026-07-29
- 许可证: CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

| 源文件（相对仓库根） | SHA-256 | 作用 |
|---|---|---|
| `attention/mla_prolog_v3/README.md` | `5628fb05be17240f1c4c18d0c9f7d22f7b76884aa27ff1e609a8326fe4ce50de` | 算子总览、公式、参数、约束 |
| `attention/mla_prolog_v3/docs/MlaPrologV3算子设计介绍.md` | `d57e211a68c30f69e12a1f797de2f86a5a079e8cba1dc0ffcdd14f10f4ddedc8` | 设计权威：四路 matmul、CV 流水 6 同步点、TilingKey 26 位、完整公式 |
| `attention/mla_prolog_v3/docs/aclnnMlaPrologV3WeightNz.md` | `d354d130e94c85658ac3e641ec48a62a12a2335d6017d0fad5ec30d31c68b532` | V3 aclnn API 定义（1657 行） |
| `attention/mla_prolog_v3/op_host/mla_prolog_v3_def.cpp` | `3df4652526c8dd37baf12f6b8a365816fcf95045b7a477fda2a6bbde4a3e427a` | 算子定义（dtype/属性/AICore config 910b/910_93/950） |
| `attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp` | `e22a2cacc755f5d67e5562406e973beb1f9b4a85861eaa06ae1ee2b34e75bc78` | kernel 入口（9 模板参数，共享 mla_prolog/ 公共 kernel） |

## 3. 关键代码区域摘要

### 3.1 计算公式（来源: README.md L28-78 / 设计介绍 L145-433）

四路计算：
- `c^Q = α_q · RmsNorm(x · W^{DQ})`（下采样）
- `q^N = c^Q · W^{UQ} · W^{UK}`（上采样 -> query）
- `q^R = ROPE(c^Q · W^{QR})`（RoPE -> query_rope）
- `k^R = ROPE(x · W^{KR})`（RoPE -> kr_cache）
- `c^{KV} = α_kv · RmsNorm(x · W^{DKV})`（下采样 -> kv_cache）

### 3.2 四路 matmul 分核策略（来源: 设计介绍 L27-45）

- MatmulCq: 沿 Hcq 按 AIC 核数切分
- MatmulCkvKr: 固定分成 64 块
- MatmulQcQr: 沿 Hqc 按 AIC 核数切分
- MatmulQn: 不切 N（M 轴单点）

### 3.3 CV 流水 6 同步点（来源: 设计介绍 L47-68 / L88-143）

MatmulCq ↔ RmsNormCq ↔ MatmulQcQr ↔ RopeQr ↔ MatmulQn 之间有 6 个 Cube-Vector 核间同步点（SYNC_MMCQ_NORMSCQ / SYNC_MMCKVKR_NORMROPE / SYNC_MMQCQR_ROPEQR / SYNC_ALL_CUBE / SYNC_ALL_VECTOR 等）。

### 3.4 TilingKey 26 位二进制（来源: 设计介绍 L70-85）

9 个字段：CACHE_MODE(0-3) / SCENARIO(4-5) / QUANT_MODE(6-9) / ENABLE_DEQUANT_OPT(10) / ENABLE_GROUP_COMPUTE_OPT(11) / EMPTY_TENSOR_MODE(12-13) / ACTUAL_SEQ_LEN_MODE(14-15) / SPLIT_M_MODE(16-17) / CV_MODE(18-25)。

### 3.5 架构分发逻辑（来源: mla_prolog_v3_def.cpp L171-172 / L315）

`ascend910b`/`ascend910_93`（arch32）与 `ascend950`（arch35，单独 dtype/layout，FLOAT8_E4M3FN 支持）。

### 3.6 V3 相对 V2 的新增特性（来源: README.md L13-27）

qcQrScale（α_q）/ kcScale（α_kv）尺度矫正因子、query_norm_flag、weightQuantMode、kvCacheQuantMode、queryQuantMode、ckvkrRepoMode、quantScaleRepoMode、tileSize（per-tile 量化）。

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用:
- `wiki/kernels/mla-prolog-v3.md`
- `artifacts/mla-prolog-v3/`

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 和关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` 并按上述路径读取原文件。tiling 参数数值、共享 kernel 实现（mla_prolog/）、per-tile 量化精度边界、cacheIndex 更新关系需结合源码/硬件进一步验证，相关结论在引用 wiki 页中标记 `confidence: inferred`。