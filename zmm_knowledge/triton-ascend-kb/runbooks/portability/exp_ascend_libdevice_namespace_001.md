---
schema_version: okf.v1
kind: constraint
type: constraint_runbook
source_family: runbooks
category: portability
dimension: 4.1
title: Triton-Ascend libdevice 命名空间为 tl.extra.ascend.libdevice，文档 xxx 占位符须按平台替换
description: 参考文档中 libdevice 路径写作 tl.extra.xxx.libdevice（占位符）；实际 910 平台为 tl.extra.ascend.libdevice。照抄占位符导致 AttributeError、全 case 编译失败；调用前以 hasattr 探测命名空间
tags: [API兼容, libdevice, 通用优化, general_ascend]
confidence: verified
status: draft
created_at: 2026-09-18T00:00:00Z
updated_at: 2026-09-18T00:00:00Z
---

## Triton-Ascend libdevice 命名空间为 tl.extra.ascend.libdevice，文档 xxx 占位符须按平台替换

【优化维度】：API 兼容
【适用条件】：平台 910 系列（实测 Ascend910_9372 / cann-9.1.0）｜ triton-ascend 全部版本 ｜ dtype 全部 ｜ shape 任意
【泛化层级】：通用（失效边界：未来 triton-ascend 版本若改命名空间，以 `import triton.language as tl; tl.extra.<tab>` 实测为准）

### 原则
- **约束**：参考文档中 libdevice 调用写作 `tl.extra.xxx.libdevice.<fn>`，`xxx` 是**占位符**，不是可运行代码
- **违反后果**：照抄 `tl.extra.cann.libdevice.tanh(z)` → `AttributeError: module 'tl.extra' has no attribute 'cann'`，kernel 编译失败，50/50 case 全挂（error_type=CompilationError 类）
- **满足方式**：910 平台固定为 `tl.extra.ascend.libdevice.<fn>`；生成代码前用运行时探测确认：
  ```python
  import triton.language as tl
  libdevice = tl.extra.ascend.libdevice  # 910 平台实测路径
  assert hasattr(libdevice, "tanh")      # 目标函数存在性一并探测
  ```

### 坏实践（反例）
```python
# ❌ 照抄文档占位符，未探测命名空间
y = 0.5 * x * (1.0 + tl.extra.cann.libdevice.tanh(z))   # AttributeError
```

### 正确写法
```python
# ✅ 平台命名空间 ascend；sigmoid 无 libdevice 版本，tanh 有
y = 0.5 * x_f32 * (1.0 + tl.extra.ascend.libdevice.tanh(z))
```

### 已知实例
- GELU（activation，Ascend910_9372）案例：`tl.extra.cann.libdevice` 致 verify 0/50（出处：`output/opt_iter_0/verify/verify_result.json.failures`），改 `tl.extra.ascend.libdevice` 后 50/50 通过（出处：`output/opt_iter_1/verify/verify_result_optimized.json`）；轨迹 `file:///home/w00836812/shixi_test02/cannbot-skills-new/plugins-official/triton-op-generator/triton_ascend_output/op_1_GELU_20260917_1827_8773/report.md`

<!-- okf:related:start -->

# 相关

- 相关主题: [逐元素算子 libdevice 超越函数 VECTOR bound 诊断链](../compute-pipeline/exp_elementwise_erf_vector_bound_001.md) — 同一轨迹（GELU），libdevice erf 诊断
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_gelu_fwd_001.md) — 消费回链：GELU fwd 案例

<!-- okf:related:end -->
