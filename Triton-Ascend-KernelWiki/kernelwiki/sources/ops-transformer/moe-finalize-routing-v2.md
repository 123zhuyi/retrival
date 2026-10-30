---
id: ops-transformer-moe-finalize-routing-v2
title: CANN ops-transformer MoeFinalizeRoutingV2 source snapshot
type: source
repo: cann/ops-transformer
path: moe/moe_finalize_routing_v2/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-31
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeFinalizeRoutingV2 来源摘要

仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（2026-07-09）。`MoeFinalizeRoutingV2` 是 MoE 计算中最后合并 FFN 输出结果的算子。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `moe/moe_finalize_routing_v2/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`

## 2. 核心源文件清单与 SHA-256

| 源文件 | SHA-256 | 作用 |
|------|------|------|
| `README.md` | `6f89e4a5a671b56a6db56633bc88d4093d73992b482c3adb5f520a123972555e` | 算子总览、公式、I/O |
| `op_host/moe_finalize_routing_v2_def.cpp` | `4b7d623f7cb5eebd58c1d794b97a1799e09f07c207091835ed3ab88e7fffb6b6` | OpDef（4 个 AICore 配置：ascend910b/ascend950/ascend310p/kirin）|
| `op_host/moe_finalize_routing_v2_tiling.h` | `d3628b9cb2abed795f8f3f304dbfaaa72563e92794ec98ad62cd21f9755648c8` | TilingData、MoeFinalizeRoutingTilingV2 基类 |
| `op_kernel/moe_finalize_routing_v2.cpp` | `4e084dd5049a32c32f13c5cebe5562ec71d29407eed1ac408613390b0083219f` | kernel 入口（~50 tiling key 分支）|

## 3. 关键代码区域摘要

### 核心公式（README.md L22-27）
```
out(i,j) = x1(i,j) + x2Optional(i,j) + Σ_k (scales(i,k) * (expandedX[expandedRowIdx(i+k*num_rows), j] + bias(expertIdx[i,k], j)))
```

### 架构支持
ascend910b/910_93（membase）、ascend950（regbase）、ascend310p、kirinX90/kirin9030

### Kernel dispatch（kernel L92-298）
~50 个 tiling key，`KERNEL_TYPE_AIV_ONLY`。分 dtype（FLOAT/FLOAT16/BF16）+ 算法策略（CutK/Db/CuthK2/CuthK4/Cuth/AllBias）+ 有/无 bias + mix_precision。