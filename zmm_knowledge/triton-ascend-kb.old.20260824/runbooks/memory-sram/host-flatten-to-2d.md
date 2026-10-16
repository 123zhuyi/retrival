---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: 多维输入必须在 host 侧扁平化为 2D
description: kernel 只处理 (row, col) 二维逻辑，多维输入 host 侧 view(prefix_dims, last_dim) 扁平化
tags: [flatten_2d, host_preprocess, multi_dim, view]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-6 多维输入 host 侧扁平化 2D

【优化维度】：内存｜访存效率
【泛化层级】：条件性（迁移条件：输入维度 >2；失效边界：输入已是 2D 无须扁平化）

<原则：kernel 内部只处理 (row, col) 二维逻辑，降低维度处理复杂度。多维输入在 host 侧 forward 中 view(prefix_dims, last_dim) 扁平化（保持 stride 不变亦可，但逻辑按 2D 处理）。与 [[normalize-dim-to-last]] 配套：先归一化目标维到末维，再扁平化为 2D。>

<骨架（占位名伪码）：
# host 侧 forward
x = x.contiguous()
prefix = x.shape[:-1]          # 所有前导维
last = x.shape[-1]
x_2d = x.view(-1, last)        # 扁平化为 2D (rows, cols)
# kernel 处理 2D
kernel[grid](x_2d, out_2d, ...)
out = out_2d.view(*prefix, last)  # 还原形状
>

**坏实践（反例）**：kernel 内处理多维
```python
# kernel 内用多维 grid + 多维坐标解码
# 维度处理复杂，坐标解码 overhead 大
```
后果：kernel 内多维逻辑复杂化，向量化差。✅ 正解即 host 侧扁平化为 2D。

**已知实例**：quantization Q5（量化算子多维输入 view 扁平化）

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 multi_dim
- 相关主题: [kernel 内用 tl.cdiv，host 侧用 triton.cdiv（禁止混用）](../portability/cdiv-kernel-vs-host.md) — 同属优化知识，主题相关
- 相关主题: [禁止在 forward() 中使用 PyTorch 计算（Type-3 退化）](../portability/no-pytorch-in-forward.md) — 主题相关：共享 host_preprocess
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 主题相关：共享 host_preprocess
- 相关主题: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](normalize-dim-to-last.md) — 主题相关：共享 host_preprocess

<!-- okf:related:end -->
