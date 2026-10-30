---
id: ops-transformer-moe-init-routing
title: CANN ops-transformer MoeInitRouting source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: moe/moe_init_routing/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-22
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeInitRouting 来源摘要

本页记录从 CANN ops-transformer 仓库 `moe/moe_init_routing/` 目录抽取的不可变来源信息。`fixture: true`。本算子为 MoE（Mixture of Experts）路由处理的第一阶段，根据 `MoeGatingTopKSoftmax` 的输出对 token 特征按 expert 排序和扩展。

## 1. 仓库与算子定位

- 仓库根：`kernelwiki/artifacts/ops-transformer/`
- 算子目录：`moe/moe_init_routing/`
- 固定提交：`d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期：2026-07-22
- 许可证：CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

完整清单见 `artifacts/moe-init-routing/source_file_inventory.md`。关键文件：

| 源文件 | SHA-256 |
|---|---|
| `README.md` | `f8172e87257a5f71...` |
| `op_host/moe_init_routing_def.cpp` | `7e87415b691a1c61...` |
| `op_kernel/moe_init_routing.cpp` | `9024dd0adb3971e0...` |
| `op_kernel/moe_common.h` | `12456104d1aa3cad...` |
| `op_kernel/moe_sort_multi_core.h` | `16452918308ebd97...` |
| `op_kernel/moe_src_to_dst_op.h` | `096f1c863a03715f...` |
| `op_kernel/moe_gather_out.h` | `19ac0dcc825fa1f6...` |

## 3. 关键代码区域摘要

### 3.1 公式（README L19-29）
`expandedExpertIdx, sortedRowIdx = keyValueSort(expertIdx, rowIdx)` → `expandedRowIdx[sortedRowIdx[i]] = i` → `expandedX[i] = x[sortedRowIdx[i] % numRows]`。

### 3.2 算子定义（def.cpp L18-72）
输入：x（FP16/BF16/FP32）、rowIdx（INT32）、expertIdx（INT32）。输出：expandedX、expandedRowIdx（INT32）、expandedExpertIdx（INT32）。属性：activeNum。AICore：ascend910b + ascend910_93 + ascend950（三架构）。

### 3.3 kernel 流水
三段流水：sort(expertIdx, rowIdx) → srcToDst（建立映射）→ gather（按映射取 expandedX）。5 tiling key（0=full load / 1-2=单/多核sort+正常gather / 3-4=单/多核sort+小活跃行gather）。`MIX_AIV_1_0`（纯 Vector，无需 Cube）。

### 3.4 产品支持
A2（910B）√、A3（910_93）√、950（950PR/950DT）√。

## 4. 衍生引用

- `wiki/kernels/moe-init-routing.md`
- `artifacts/moe-init-routing/`

## 5. 复现性说明

本页仅记录源文件路径、SHA-256 与关键代码区域摘要。要复现分析需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。