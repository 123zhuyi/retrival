---
id: ops-transformer-mla-prolog
title: CANN ops-transformer MlaProlog source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/mla_prolog/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MlaProlog 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/mla_prolog/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。`fixture: true`。支持 A2/A3 + Ascend950（FP8）。

## 1. 仓库与算子定位

- 仓库根：`kernelwiki/artifacts/ops-transformer/`
- 算子目录：`attention/mla_prolog/`
- 固定提交：`d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期：2026-07-29
- 许可证：CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

完整清单见 `artifacts/mla-prolog/source_file_inventory.md`。关键文件 SHA-256：

| 源文件（相对算子目录） | SHA-256 |
|---|---|
| `README.md` | `bd0059922d16b0f88450e2736d6fc53f206c4c0ed69fe4a84b555e51f65f581b` |
| `docs/MlaProlog算子设计介绍.md` | `2c079807b5efafcac7a593a44e51154b862a349804056628c5d21caf39cd57d7` |
| `docs/aclnnMlaProlog.md` | `228c925df2d06b732d8cb39f8d4b65f179a5060c5054746b3223368a39d668a0` |
| `op_host/mla_prolog_def.cpp` | `d9d8661e9b5c8fa52d41aa313500ea4b4e28780898af9705c1e5961fd09084ce` |
| `op_host/mla_prolog_tiling.cpp` | `f62cfa3324e05ad3eadddb2864a41c80ae962563d8f24d930911a6ce90eff3ad` |
| `op_host/mla_prolog_tiling.h` | `fb5b3d6784a52a1837a95166df20949fe2444bac932c98be77596fa6accf3e4a` |
| `op_kernel/mla_prolog.cpp` | `2309ef39ee78bc47cb1bdbc7d082cfe965e99143cbd50eb2a37f2943aa8c1bc3` |
| `op_kernel/kernel_mla_prolog_split_n.h` | `cc9e9ccfa531f1fd123939b1cb44649a9d5251c1e7adf8fb6fc60d60d8201150` |
| `op_kernel/mla_prolog_template_tiling_key.h` | `1188bed8af1320cd80e13f424e7a9baa13094f30ea1a47a1f61152ab898d8867` |
| `op_kernel/service_matmul.h` | `e951e748aaa9cef3e9a81c4bed8eaa55d395445880f157eca40ec714666bf707` |
| `op_kernel/service_rms_norm.h` | `fddf4682c65c9d5b8b97172b23f22e563975c5c3894364c780dd216cd83e21f4` |
| `op_kernel/service_scatter_cache.h` | `3e4385b4de756919b49146cb050d6948e47a511a81c9b9d518d4dbc95ace14be` |

## 3. 关键代码区域摘要

### 3.1 四路数据流（README L12-60；设计介绍 L9-22）
x → ×W^DQ(RmsNorm)→[×W^UQ·W^UK→q^N] + [×W^QR→RoPE→q^R] + [×W^DKV→RmsNorm→cache→k^C] + [×W^KR→RoPE→cache→k^R]。权重 W^UQ+W^QR 与 W^DKV+W^KR 分别横向拼接为一个 matmul 后 split（设计介绍 L269-273）。

### 3.2 算子定义（def.cpp L23-136/267）
- 输入：token_x/weight_dq/weight_uq_qr/weight_uk/weight_dkv_kr/rmsnorm_gamma_cq/ckv/rope_sin/cos/cache_index/kv_cache/kr_cache（多为 REQUIRED）+ 可选 dequant/quant/smooth_scales。
- 输出：query/query_rope/kv_cache/kr_cache。
- 属性：rmsnorm_epsilon_cq/ckv(1e-5)、cache_mode("BNSD")。
- AICore config：ascend910b/910_93(BF16/INT8) + ascend950(额外 FP8_E4M3FN)。

### 3.3 tiling key 二进制编码（设计介绍 L70-79）
CACHE_MODE(2bits) + SCENARIO(2bits) + QUANT_MODE(4bits) + ENABLE_DEQUANT_OPTIONAL(1bit) + ENABLE_GROUP_COMPUTE_OPTIONAL(1bit) + EMPTY_TENSOR_MODE(2bits)。kernel 入口 8 模板参数（mla_prolog.cpp L20-22）。3 场景：NO_QUANT / PARTIAL_QUANT_KV_NO_QUANT / PARTIAL_QUANT_KV_QUANT_PER_CHANNEL（L69-99）。

### 3.4 CV 流水（设计介绍 L47-67/Process 伪代码 L83-146）
6 个 CV sync event：SYNC_MMCQ_NORMSCQ / SYNC_MMCKVKR_NORMROPE / SYNC_NOMRCQ_MMQCQR / SYNC_MMQCQR_ROPEQR / SYNC_ALL_CUBE / SYNC_ALL_VECTOR。Cube 做 4 matmul（MatmulCq/CkvKr/QcQr/Qn），Vector 做 RmsNorm/RoPE/Cache scatter。

### 3.5 4 matmul 切分（设计介绍 L36-44）
MatmulCq(BS×He→Hcq) + MatmulCkvKr(BS×He→Hckv+Dr) + MatmulQcQr(BS×Hcq→N*(D+Dr)) + MatmulQn(BS×N*D→Hckv)。MIX_AIC_1_2 CV 配比。

## 4. 衍生引用

- `wiki/kernels/mla-prolog.md`
- `artifacts/mla-prolog/`

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 与关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。tiling 参数数值、FP8 精度边界、MLA 权重吸收实现、预留参数（未使用）等需结合硬件/源码进一步验证，相关结论在引用 wiki 页中标记 `confidence: inferred`。