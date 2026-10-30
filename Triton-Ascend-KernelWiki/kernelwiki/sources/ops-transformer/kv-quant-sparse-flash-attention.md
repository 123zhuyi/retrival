---
id: ops-transformer-kv-quant-sparse-flash-attention
title: CANN ops-transformer KvQuantSparseFlashAttention source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/kv_quant_sparse_flash_attention/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer KvQuantSparseFlashAttention 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/kv_quant_sparse_flash_attention/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。`fixture: true`。KV Quant SFA 是 `sparse_flash_attention` 的 KV 量化版本（Per-Token-Head-Tile-128 量化），仅推理场景。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `attention/kv_quant_sparse_flash_attention/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期: 2026-07-29
- 许可证: CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

| 源文件（相对仓库根） | SHA-256 | 作用 |
|---|---|---|
| `attention/kv_quant_sparse_flash_attention/README.md` | `ef0c7ddcb217b0a6d1c2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3` | 算子总览、公式、参数、约束 |
| `attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_def.cpp` | `f9b3a97047197616f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3` | 算子定义（dtype/属性/AICore config） |
| `attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.cpp` | `d5ec7a3dea4dd0d4e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3` | tiling 主逻辑 |
| `attention/kv_quant_sparse_flash_attention/op_kernel/kv_quant_sparse_flash_attention.cpp` | `1cd8024ff0a89b11a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4` | kernel 入口（4 模板参数） |
| `attention/kv_quant_sparse_flash_attention/op_kernel/kv_quant_sparse_flash_attention_kernel_mla.h` | `0f52f705d4597095f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3` | arch32 kernel 核心（MLA-absorb） |
| `attention/kv_quant_sparse_flash_attention/op_kernel/arch35/kv_quant_sparse_flash_attention_kernel_mla.h` | `866dc0a14e6a0a8ee1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3` | arch35 kernel 核心（FP8_E4M3FN/HIFLOAT8） |
| `attention/kv_quant_sparse_flash_attention/op_kernel/kv_quant_sparse_flash_attention_template_tiling_key.h` | `63f161d643a30e6ef1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3` | 模板参数声明（4 字段，5 组合） |

## 3. 关键代码区域摘要

### 3.1 计算公式（来源: README.md L14-20）

```
Attention = softmax((Q · Dequant(K^INT8, Scale_K)^T) / √d) · Dequant(V^INT8, Scale_V)
```

### 3.2 模板参数与 dispatch（来源: kv_quant_sparse_flash_attention.cpp L37-73 / template_tiling_key.h L31-37）

4 个模板参数：FLASH_DECODE(bool) / LAYOUT_T(BSND/TND) / KV_LAYOUT_T(BSND/TND/PA_BSND) / TEMPLATE_MODE(C/V)。5 种合法组合。arch35（950）分支独立处理 FP8_E4M3FN/HIFLOAT8。

### 3.3 MLA-absorb 模式（来源: README.md L94-95）

attention_mode=2，Q/K D 维包含 nope+rope 拼接（Q: 512+64=576，K: 512+64*2+4*4=656）。KV 量化参数和数据混合存放（quant_scale_repo_mode=1）。

### 3.4 架构分发逻辑（来源: kv_quant_sparse_flash_attention_def.cpp L92-93/L151）

`ascend910b`（A2 推理）/ `ascend910_93`（A3 推理）/ `ascend950`（950，FP8_E4M3FN/HIFLOAT8 支持）。仅推理场景。

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用:
- `wiki/kernels/kv-quant-sparse-flash-attention.md`
- `artifacts/kv-quant-sparse-flash-attention/`

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 和关键代码区域摘要，不包含完整源码。SHA-256 值为示意（前 16 位），完整值需在固定提交下重算。tiling 参数数值、per-tile 量化精度边界、sparse_indices 有效值约定需结合源码/硬件进一步验证，相关结论在引用 wiki 页中标记 `confidence: inferred`。