---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.3"
title: 动态读取 Vector Core 数量，禁止硬编码
description: kernel 启动前运行时读取实际 Vector Core 数量作为 grid 上限，禁止硬编码 num_cores
tags: [vector_core, num_cores, dynamic, grid_cap]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-1 动态读取 Vector Core 数量

【优化维度】：计算｜多核负载
【泛化层级】：Triton通用（迁移条件：所有 Ascend NPU kernel；无失效边界）

<原则：硬编码 num_cores 仅利用部分 Vector Core（如硬编码 8 时仅用 20%），导致加速比从 ~1.3x 跌至 0.67x（慢于 PyTorch）。必须运行时读取实际核数。与 [[grid-cap-to-core-num]] 配套：grid 上限取动态核数。>

<骨架（占位名伪码）：
VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get('vector_core_num', 40)
# 或 triton.runtime.driver.active.utils.get_device_properties(device)['num_vectorcore']
grid = (min(total_blocks, VEC_CORE_NUM),)
>

**坏实践（反例）**：硬编码核数
```python
grid = (min(total_blocks, 48),)   # 硬编码 48
```
后果：在某些芯片上仅利用 20% Vector Core，加速比 1.3x 跌至 0.67x。✅ 正解即本条动态读取。

**已知实例**：tensor-transform G1（`triton-op-generator/template/tensor-transform.md`）

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](../../ops/moe-finalize-routing-v2-grad.md) — 主题相关：共享 vector_core
- 实践案例: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](../../ops/moe-finalize-routing-v2.md) — 主题相关：共享 vector_core
- 实践案例: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](../../ops/moe-gating-top-k-softmax-v2.md) — 主题相关：共享 vector_core
- 实践案例: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](../../ops/moe-init-routing-v2-grad.md) — 主题相关：共享 vector_core
- 实践案例: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](../../ops/moe-init-routing-v3.md) — 主题相关：共享 vector_core
- 实践案例: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](../../ops/moe-token-permute-grad.md) — 主题相关：共享 vector_core
- 实践案例: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](../../ops/moe-token-permute-with-ep-grad.md) — 主题相关：共享 vector_core
- 实践案例: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](../../ops/moe-token-permute-with-routing-map.md) — 主题相关：共享 vector_core
- 实践案例: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](../../ops/moe-token-unpermute-with-ep.md) — 主题相关：共享 vector_core
- 实践案例: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 vector_core
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../autotuning/autotune-num-warps-stages.md) — 同属优化知识，主题相关
- 相关主题: [matmul core mapping 策略选择（列/行/对角）](core-mapping-strategy-selection.md) — 同属优化知识，主题相关
- 相关主题: [固定 CUBE_CORE_NUM 启动 + 核内循环（Cube 类算子）](cube-core-num-fixed.md) — 同属优化知识，主题相关
- 相关主题: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](diagonal-core-mapping-l2-bank-conflict.md) — 同属优化知识，主题相关
- 相关主题: [按输出元素数（而非输入）分配 grid，且钳制到 num_cores](grid-by-output-count.md) — 同属优化知识，主题相关
- 相关主题: [grid 总数不超过核数](grid-cap-to-core-num.md) — 主题相关：共享 grid_cap, num_cores
- 相关主题: [grid_size 必须与 num_cores 入参严格相等](grid-strict-match-num-cores.md) — 主题相关：共享 num_cores
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](load-balance-formula.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
