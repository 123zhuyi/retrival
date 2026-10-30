---
id: ops-transformer-moe-init-routing-v2
title: CANN ops-transformer MoeInitRoutingV2 source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: moe/moe_init_routing_v2/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-22
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeInitRoutingV2 来源摘要

MoE 路由 v2。与 v1 差异：无 rowIdx 输入，新增 expert_capacity/drop_pad_mode/expert_num 属性，新增 expert_tokens_count/expert_tokens_before_capacity 输出，新增 ascend310p 支持。

## 1. 仓库与算子定位

- 仓库根：`kernelwiki/artifacts/ops-transformer/`
- 算子目录：`moe/moe_init_routing_v2/`
- 固定提交：`d28d8f99a8858b71caa27a535da32bb5ba68ce2a`

## 2. 核心源文件清单

完整清单见 `artifacts/moe-init-routing-v2/source_file_inventory.md`。关键文件：

| 源文件 | SHA-256 |
|---|---|
| `op_host/moe_init_routing_v2_def.cpp` | `1e687da3431b82f3...` |
| `op_kernel/moe_init_routing_v2.cpp` | `122b1b6a7875178d...` |
| `op_kernel/moe_v2_src_to_dst_with_capacity.h` | `66094b6dc0d50a1e...` |
| `op_kernel/moe_v2_expert_token_out.h` | `a8b3960a58c28be1...` |

## 3. 关键差异（vs v1）

- 输入：仅 x + expertIdx（无 rowIdx）
- 新增属性：expert_capacity、expert_num、drop_pad_mode、expert_tokens_count_or_cumsum_flag、expert_tokens_before_capacity_flag
- 新增可选输出：expert_tokens_count_or_cumsum、expert_tokens_before_capacity
- 新增架构：ascend310p
- 三段流水 + expert token 计数输出

## 4. 衍生引用

- `wiki/kernels/moe-init-routing-v2.md`
- `artifacts/moe-init-routing-v2/`