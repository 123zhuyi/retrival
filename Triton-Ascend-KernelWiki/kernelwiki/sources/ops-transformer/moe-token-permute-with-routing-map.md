---
id: ops-transformer-moe-token-permute-with-routing-map
title: CANN ops-transformer MoeTokenPermuteWithRoutingMap source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: moe/moe_token_permute_with_routing_map/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeTokenPermuteWithRoutingMap 来源摘要

本页记录从 CANN ops-transformer 仓库 `moe/moe_token_permute_with_routing_map/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。`fixture: true`。MoE Token Permute With Routing Map 是 routingMap 驱动的 MoE token 置换算子。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `moe/moe_token_permute_with_routing_map/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期: 2026-07-29
- 许可证: CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

| 源文件（相对仓库根） | SHA-256 | 作用 |
|---|---|---|
| `moe/moe_token_permute_with_routing_map/README.md` | `3feb68e3521003b3a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3` | 算子总览、公式（2 种模式）、参数 |
| `moe/moe_token_permute_with_routing_map/op_host/moe_token_permute_with_routing_map_def.cpp` | `ad76413e435fbc72f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3` | 算子定义（AICore config 910b/910_93/950） |
| `moe/moe_token_permute_with_routing_map/op_kernel/moe_token_permute_with_routing_map.cpp` | `601cfa228160484ce1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3` | kernel 入口（TilingKey 1-9/19 分发） |
| `moe/moe_token_permute_with_routing_map/op_kernel/masked_select_v3.h` | `7e1fcb567f56b001f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3` | maskedSelect 核心实现 |
| `moe/moe_token_permute_with_routing_map/op_kernel/moe_sort_multi_core.h` | `d08beddd5fc473d8f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3` | 多核排序 |

## 3. 关键代码区域摘要

### 3.1 计算公式（来源: README.md L18-89）

两种模式：
- dropAndPad=false：maskedSelect + argSort + `permuteTokens[sortedIndicesOut[i]] = tokens[i//topK]`
- dropAndPad=true：capacity = numOutTokens // expert_num，`permutedTokensOut = tokens.index_select(0, sortedIndicesOut)`

### 3.2 TilingKey 分发（来源: moe_token_permute_with_routing_map.cpp L99-154）

- 1/3/5/7：单核排序（MoeSortOneCore）
- 2/4/6/8：多核排序（MoeSortMultiCore/MoeSortMultiLastDimCore + moe_mrgsort 归并）
- 9/19：填充模式（仅 3510/arch35）

### 3.3 架构分发（来源: def.cpp L80-88）

`ascend910b`/`ascend910_93`（arch32 membase）/ `ascend950`（arch35 regbase + SIMD/SIMT gather）。

### 3.4 核心模板（来源: kernel 头文件）

`KernelMaskedSelectV3`（maskedSelect）、`MoeSortOneCore`/`MoeSortMultiCore`/`MoeSortMultiLastDimCore`（排序）、`MoeindexCopyOp`/`MoeindexCopySplitDOp`（index_copy）、`MoePermuteProb`（probs 置换）、`Gatherv2Simd/SimtTwoDim`（arch35 二维 gather）。

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用:
- `wiki/kernels/moe-token-permute-with-routing-map.md`
- `artifacts/moe-token-permute-with-routing-map/`

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 和关键代码区域摘要。SHA-256 值为示意（前 16 位），完整值需在固定提交下重算。tiling 参数数值、多核归并排序映射、arch35 SIMD/SIMT gather 精度需结合源码/硬件进一步验证。