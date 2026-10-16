---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: 3.3
title: sqrt 前用 max(x, eps) 钳位防负数开方 NaN（eps 按精度选型）
description: 910b2/910b3 上硬件 sqrt 对负数输入直接返回 NaN，开方前用 tl.maximum(x, eps) 钳位——fp16/bf16 建议 eps=1e-6、fp32 建议 1e-8，解决数值误差产生的微小负数开方问题
tags: [数值稳定, aiv, general_stability]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## sqrt 前用 max(x, eps) 钳位防负数开方 NaN（eps 按精度选型）

【优化维度】：数值稳定
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：钳位会把合法的 0 或微小正值抬高到 eps——输出 sqrt(eps)≈1e-3（eps=1e-6 时）替代真实小值，对输出绝对精度敏感的场景须用 0.0 钳位或评估 eps；真实输入本就可能为负（非数值误差）时钳位会掩盖 bug，应先确认负数只来自浮点误差；精度收益与性能代价：收益=消除 NaN，代价=一次 max 指令（可忽略）+ 小值域精度让步；MERE/MARE/ULP 误差指标未测定）

### 原则
- **核心策略**：平方根操作前用max函数限制被开方数最小值（0或极小eps），避免负数开方
- **eps选择**：低精度（fp16/bf16）建议用1e-6，高精度（fp32）建议用1e-8，平衡精度与稳定性
- **场景适配**：专门解决数值误差导致的微小负数（如-1e-10）开方返回NaN的问题

### 代码/骨架出处
- `Triton`:
```python
@triton.jit def sqrt_negative_opt_kernel(input_ptr, output_ptr, n_elements, eps: tl.constexpr=1e-6, blockSize: tl.constexpr=128):
    pid = triton.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    mask = offsets < n_elements
    
    # 加载数据
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 核心优化: 确保被开方数非负
    safe_data = tl.maximum(data, eps)  # 替代方案: tl.maximum(data, 0.0)
    
    # 平方根计算（无NaN风险）
    sqrt_data = tl.sqrt(safe_data)
    
    # 存储结果
    tl.store(output_ptr + offsets, sqrt_data, mask=mask)

# Attention的sqrt(d_k)场景示例
def attention_sqrt_opt(d_k):
    # 避免d_k为0或负数（极端场景）
    safe_d_k = max(d_k, 1e-6)
    sqrt_d_k = tl.sqrt(tl.cast(safe_d_k, tl.float32))
    return sqrt_d_k

# 通用平方根调用
def sqrt_opt(input_tensor):
    input_contig = input_tensor.contiguous() if not input_tensor.is_contiguous() else input_tensor
    output = torch.empty_like(input_contig)
    n_elements = input_contig.numel()
    grid = (triton.cdiv(n_elements, 128),)
    sqrt_negative_opt_kernel[grid](input_contig, output, n_elements, eps=1e-6, blockSize=128)
    return output
```

**坏实践（反例）**：对浮点误差产生的微小负数（如方差算出 -1e-10）直接 `tl.sqrt` → 910b2/910b3 硬件 sqrt 对负数直接返回 NaN 并向后传播 → ✅ 正解即本条：`tl.maximum(x, eps)` 前置钳位，eps 按精度选型（fp16/bf16 用 1e-6，fp32 用 1e-8）；适用 attention 的 sqrt(d_k)、均方差、梯度平方根等场景

**已知实例**：general_stability §raw (`experiences/general_stability/exp_stability_002.yaml`)
- 声称性能收益：数值稳定性提升95%
- 备注：适用于Attention的sqrt(d_k)、均方差计算、梯度平方根等场景；910b2/910b3平台的硬件sqrt指令对负数输入直接返回NaN，必须前置该防护逻辑

<!-- okf:related:start -->

# 相关

- 开发指南: [归约算子优化](../../reference/op-class-reduction.md) — 同属优化知识，主题相关
- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](../memory-sram/exp_perf_009.md) — 主题相关：共享 aiv
- 相关主题: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](exp_stability_001.md) — 主题相关：共享 aiv, general_stability, 数值稳定
- 相关主题: [关键计算步骤转 fp32 执行再转回原精度（MAE 降 80%，算力代价约 5%）](exp_stability_003.md) — 主题相关：共享 aiv, general_stability
- 相关主题: [累加器必须 fp32 初始化：fp16 累加 ≥1000 次漂移明显（MSE 降 90%）](exp_stability_004.md) — 主题相关：共享 aiv, general_stability

<!-- okf:related:end -->
