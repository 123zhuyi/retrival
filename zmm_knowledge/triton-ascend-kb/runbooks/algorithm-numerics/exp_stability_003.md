---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: 3.2
title: 关键计算步骤转 fp32 执行再转回原精度（MAE 降 80%，算力代价约 5%）
description: 910b2/910b3 上多层激活/累积求和等复杂计算，中间步骤转 float32 执行、完成后转回 fp16/bf16 存储；声称 MAE 降 80%+，fp32 计算耗时仅高 5%，精度收益远大于性能损耗
tags: [混合精度, aiv, general_stability]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 关键计算步骤转 fp32 执行再转回原精度（MAE 降 80%，算力代价约 5%）

【优化维度】：混合精度
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：访存受限（memory-bound）kernel 转 fp32 的收益主要是计算精度，若瓶颈在带宽则性能代价近零但精度收益不变；计算受限且本身已在 CUBE 单元用 fp32 累加的矩阵乘无需重复转换；5% 算力代价是 910b2/910b3 实测量级，逐 element 计算占比极高的 kernel 代价会放大；精度收益与性能代价：MAE 降 80%（声称值，具体 MERE/MARE 依算子而定）vs 算力 +5%；ULP 级误差指标未测定）

### 原则
- **核心策略**：关键计算步骤转为float32执行，完成后转回原精度（fp16/bf16）存储
- **精度提升**：fp16转float32计算可将MAE（平均绝对误差）降低80%以上
- **性能平衡**：910b2/910b3的float32计算耗时仅比fp16高5%，精度收益远大于性能损耗

### 代码/骨架出处
- `Triton`:
```python
@triton.jit def mid_float32_opt_kernel(x_ptr, y_ptr, output_ptr, M, N, K, blockM: tl.constexpr=256, blockN: tl.constexpr=256, blockK: tl.constexpr=256):
    pid = triton.program_id(0)
    rm = pid // (N // blockN)
    rn = pid % (N // blockN)
    
    # 加载低精度数据
    x_ptr = x_ptr + rm * blockM * K + tl.arange(0, blockM)[:, None] * K + tl.arange(0, blockK)[None, :]
    y_ptr = y_ptr + rn * blockN + tl.arange(0, blockK)[:, None] * N + tl.arange(0, blockN)[None, :]
    x = tl.load(x_ptr, mask=tl.arange(0, blockK)[None, :] < K, other=0.0)
    y = tl.load(y_ptr, mask=tl.arange(0, blockK)[:, None] < K, other=0.0)
    
    # 核心优化: 转为float32进行中间计算
    x_fp32 = x.to(tl.float32)
    y_fp32 = y.to(tl.float32)
    acc = tl.zeros((blockM, blockN), dtype=tl.float32)
    
    # float32高精度计算
    acc += tl.matmul(x_fp32, y_fp32)
    
    # 转回原精度存储
    acc_fp16 = acc.to(x.dtype)
    output_ptr = output_ptr + rm * blockM * N + tl.arange(0, blockM)[:, None] * N + tl.arange(0, blockN)[None, :]
    tl.store(output_ptr, acc_fp16, mask=tl.arange(0, blockM)[:, None] < M - rm * blockM and tl.arange(0, blockN)[None, :] < N - rn * blockN)
```

**坏实践（反例）**：长链计算全程 fp16/bf16（中间结果不转 fp32）→ 每步舍入误差沿链路累积放大，多层激活/连乘场景输出 MAE 显著恶化 → ✅ 正解即本条：关键中间步骤 `.to(tl.float32)` 计算、最终转回原精度存储；注意全程 fp32 也无必要——IO 保持低精度可省一半带宽

**已知实例**：general_stability §raw (`experiences/general_stability/exp_stability_003.yaml`)
- 声称性能收益：计算精度提升（MAE降低80%）
- 备注：适用于多层激活、累积求和等复杂计算场景；fp16/bf16的动态范围有限，中间计算易丢失小数位，910b2/910b3平台建议所有核心计算步骤启用float32转换

<!-- okf:related:start -->

# 相关

- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](../memory-sram/exp_perf_009.md) — 主题相关：共享 aiv
- 相关主题: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](exp_stability_001.md) — 主题相关：共享 aiv, general_stability
- 相关主题: [sqrt 前用 max(x, eps) 钳位防负数开方 NaN（eps 按精度选型）](exp_stability_002.md) — 主题相关：共享 aiv, general_stability
- 相关主题: [累加器必须 fp32 初始化：fp16 累加 ≥1000 次漂移明显（MSE 降 90%）](exp_stability_004.md) — 主题相关：共享 aiv, general_stability, 混合精度

<!-- okf:related:end -->
