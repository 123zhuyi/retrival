---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: pertensor scale 必须物化为连续 [1,N]，禁止 stride-0 广播 load
description: pertensor scale=[1] 经 expand 后必须 contiguous 物化为 [1,N]，stride-0 视图会触发 UB overflow
tags: [pertensor_scale, materialize, stride_zero, ub_overflow, broadcast_load]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-7 pertensor scale 物化连续

【优化维度】：内存｜访存效率
【泛化层级】：条件性（迁移条件：pertensor 量化 scale=[1] 需广播到 [1,N]；失效边界：pergroup/perchannel scale 已是连续张量）

<原则：pertensor scale=[1] 经 expand 得 stride-0 视图，Ascend 编译器无法 tile，触发 UB overflow（实测 790KB > 192KB UB）。必须 contiguous 物化为 [1,N] 走正常向量 load。>

<骨架（占位名伪码）：
# pertensor scale=[1] -> 物化为连续 [1,N]
scale_mat = scale.view(1, 1).expand(1, N).contiguous()   # stride (N,1)，连续
# kernel 内正常向量 load
s = tl.load(scale_ptr + offs_n)
>

**坏实践（反例）**：stride-0 广播 load
```python
scale_mat = scale.view(1, 1).expand(1, N)   # stride-0 视图，未 contiguous
# kernel 内 tl.load(scale_ptr + offs_n * 0)  # 编译器无法 tile
```
后果：UB overflow（790KB > 192KB），iter_0 教训。✅ 正解即 contiguous 物化。

**已知实例**：weight-quant-matmul W3（iter_0 UB overflow 教训）

<!-- okf:related:start -->

# 相关

- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 主题相关：共享 ub_overflow

<!-- okf:related:end -->
