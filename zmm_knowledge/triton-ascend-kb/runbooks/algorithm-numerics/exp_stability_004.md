---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: 3.2
title: 累加器必须 fp32 初始化：fp16 累加 ≥1000 次漂移明显（MSE 降 90%）
description: 910b2/910b3 上大步数累加/梯度累积/损失求和场景，累加器用 tl.zeros(..., dtype=tl.float32) 初始化；fp16 累加器 1000 次累加误差达 1e-3，fp32 仅 1e-7，声称 MSE 降 90%
tags: [混合精度, aiv, general_stability]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 累加器必须 fp32 初始化：fp16 累加 ≥1000 次漂移明显（MSE 降 90%）

【优化维度】：混合精度
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：累加步数少（远小于 1000）或输入动态范围小时 fp16 累加误差可接受，可不强转；累加值本身可能超 fp32 范围（极长累加链 + 大数值）时须改用 Kahan 求和或分块归约；精度收益与性能代价：MSE 降 90%（声称值，fp16 累加 1000 步误差 1e-3 vs fp32 1e-7）vs 累加器寄存器占用翻倍 + 每步转换指令；ULP 级误差指标未测定）

### 原则
- **核心策略**：累加器初始化为float32高精度类型，存储中间累加结果
- **防漂移**：低精度累加器（fp16）在≥1000次累加后会出现明显精度漂移，float32累加器可完全避免

### 代码/骨架出处
- `Triton`:
```python
@triton.jit def high_precision_acc_kernel(input_ptr, output_ptr, n_elements, blockSize: tl.constexpr=128):
    pid = triton.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    mask = offsets < n_elements
    
    # 加载数据
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 核心优化: 初始化float32高精度累加器
    acc_fp32 = tl.zeros((blockSize,), dtype=tl.float32)
    
    # 高精度累加（模拟多步累加场景）
    for i in range(1000):  # 示例: 1000次累加
        acc_fp32 += data.to(tl.float32) * (i + 1)  # 累加不同权重的数值
    
    # 转回原精度存储
    acc_fp16 = acc_fp32.to(data.dtype)
    tl.store(output_ptr + offsets, acc_fp16, mask=mask)

# 累加求和调用示例
def sum_high_precision(input_tensor):
    input_contig = input_tensor.contiguous() if not input_tensor.is_contiguous() else input_tensor
    output = torch.empty_like(input_contig)
    n_elements = input_contig.numel()
    grid = (triton.cdiv(n_elements, 128),)
    high_precision_acc_kernel[grid](input_contig, output, n_elements, blockSize=128)
    return output.sum()
```

**坏实践（反例）**：累加器沿用输入 dtype（`acc = tl.zeros(..., dtype=tl.float16)`）跑长累加链 → ≥1000 步后精度漂移肉眼可见（误差达 1e-3），梯度累积/损失求和直接失真 → ✅ 正解即本条：`tl.zeros(..., dtype=tl.float32)` 初始化累加器，每步 `.to(tl.float32)` 再累加，最终转回

**已知实例**：general_stability §raw (`experiences/general_stability/exp_stability_004.yaml`)
- 声称性能收益：累加精度提升（MSE降低90%）
- 备注：适用于大步数累加、梯度累积、损失求和等场景；fp16累加器在1000次累加后误差可达1e-3，float32累加器误差仅1e-7，910b2/910b3平台建议所有累加操作启用高精度累加器

<!-- okf:related:start -->

# 相关

- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](../memory-sram/exp_perf_009.md) — 主题相关：共享 aiv
- 相关主题: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](exp_stability_001.md) — 主题相关：共享 aiv, general_stability
- 相关主题: [sqrt 前用 max(x, eps) 钳位防负数开方 NaN（eps 按精度选型）](exp_stability_002.md) — 主题相关：共享 aiv, general_stability
- 相关主题: [关键计算步骤转 fp32 执行再转回原精度（MAE 降 80%，算力代价约 5%）](exp_stability_003.md) — 主题相关：共享 aiv, general_stability, 混合精度

<!-- okf:related:end -->
