---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: 输入进入 kernel 前必须 contiguous
description: host 侧 forward 中调用 x.contiguous()，避免非连续张量导致 kernel 内 stride 计算复杂化
tags: [contiguous, stride, memory_access, host_preprocess]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-2 输入 contiguous 化

【优化维度】：内存｜访存效率
【泛化层级】：Triton通用（迁移条件：所有 kernel 输入张量；无失效边界）

<原则：非连续张量在 kernel 内需要额外 stride 计算和分散访存，破坏 GM->UB 连续 coalescing。host 侧 forward 进入 kernel 前统一调用 .contiguous() 保证连续访存。>

<骨架（占位名伪码）：
# host 侧 forward
def forward(self, input_tensor, ...):
    input_tensor = input_tensor.contiguous()   # 进入 kernel 前保证连续
    kernel[grid](input_tensor, ...)
>

**坏实践（反例）**：直接传非连续张量给 kernel
```python
def forward(self, x, ...):
    kernel[grid](x, ...)   # x 可能是非连续（如 transpose 后）
# kernel 内需要 stride 计算，访存分散
```
后果：kernel 内 stride 计算复杂化，破坏连续 coalescing，GM->UB 访存效率下降。✅ 正解即 host 侧 .contiguous()。

**已知实例**：tensor-transform G7（`triton-op-generator/template/tensor-transform.md`）

<!-- okf:related:start -->

# 相关

- 相关主题: [grid_size 必须与 num_cores 入参严格相等](../compute-pipeline/grid-strict-match-num-cores.md) — 主题相关：共享 stride
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — stride 匹配布局检查项的 host 侧前置
- 相关主题: [禁止在 forward() 中使用 PyTorch 计算（Type-3 退化）](../portability/no-pytorch-in-forward.md) — 主题相关：共享 host_preprocess
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](exp_perf_006.md) — 同属优化知识，主题相关
- 相关主题: [多维输入必须在 host 侧扁平化为 2D](host-flatten-to-2d.md) — 主题相关：共享 host_preprocess
- 相关主题: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](normalize-dim-to-last.md) — 主题相关：共享 host_preprocess

<!-- okf:related:end -->
