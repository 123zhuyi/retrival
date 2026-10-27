---
title: 调试、稳定性与 API 规范
category: workflows
tags: [debug, stability, numerical, fp16, overflow, mask, api-rules]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# 调试、稳定性与 API 规范

> 本页汇总调试排查、数值稳定性和 API 语法规范，来自 `raw/experiences/general_debug/`（5 条）、`general_stability/`（4 条）、`general_api/`（4 条）共 13 条经验。

---

## Part 1：调试排查

### 1.1 内存访问三项检查

**每次 load/store 必须检查**：

| 检查项 | 说明 | 修复 |
|--------|------|------|
| mask 检查 | 所有 `tl.load/tl.store` 必须带 mask 或 boundary_check | 添加 `mask = offsets < n` |
| stride 检查 | stride 参数必须与张量实际内存布局匹配 | 使用 `tensor.stride(dim)` 传入正确 stride |
| 索引越界 | 所有数组索引需条件判断（`offset < n`） | 加 mask 后再 load |

```python
# 正确
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
data = tl.load(ptr + offsets * stride, mask=mask)  # stride 匹配实际内存布局
tl.store(out_ptr + offsets * stride, data, mask=mask)

# 错误
data = tl.load(ptr + offsets * stride)   # 无 mask，可能越界
data = tl.load(ptr + offsets * 2, ...)   # stride=2 写死，布局改变后出错
```

经验文件：[exp_debug_001](../../raw/experiences/general_debug/exp_debug_001.yaml)

### 1.2 精度对比调试

精度调试标准步骤：

```python
# 1. 用 fp32 参考实现验证数学正确性
ref = reference_impl(x.float())

# 2. 运行 Triton kernel
out = triton_kernel(x)

# 3. 检查相对误差
max_err = (out - ref).abs().max() / ref.abs().max()
print(f"Max relative error: {max_err:.6f}")  # 应 < 1e-3 (fp16) / 1e-5 (fp32)

# 4. 如果误差大，尝试升精度中间计算
# 在 kernel 内将关键计算升 fp32：
acc = tl.zeros([BLOCK], dtype=tl.float32)
data = tl.load(ptr + offsets, mask=mask).to(tl.float32)  # 升精度
# ... 计算 ...
tl.store(out_ptr + offsets, acc.to(tl.float16), mask=mask)  # 降精度存储
```

经验文件：[exp_debug_002](../../raw/experiences/general_debug/exp_debug_002.yaml)

### 1.3 shape/dtype 不匹配排查

常见错误：传入 kernel 的 shape/stride 与实际 tensor 不匹配。  
排查步骤：
1. 打印 `tensor.shape, tensor.stride(), tensor.dtype` 确认
2. 检查是否在 host 侧做了 `.contiguous()` 但忘记更新 stride 参数
3. 非连续 tensor（如 transpose 后未 contiguous）的 stride 不是 1

经验文件：[exp_debug_003](../../raw/experiences/general_debug/exp_debug_003.yaml)

### 1.4 编译错误快速定位

| 错误类型 | 常见原因 | 解决方案 |
|----------|---------|----------|
| `tl.constexpr` 相关错误 | host 侧使用了 tl.constexpr | 改为普通 Python 变量 |
| Grid 乘积超限 | x * y * z > 65535 | 减小 grid 维度，改用 grid-stride loop |
| UB 超限 | tile 大小 × dtype_size > 192KB | 减小 BLOCK_M/N/K |
| 维度对齐错误 | 尾轴步长非 512B 倍数 | 调整 stride 对齐 |
| NaN/Inf 输出 | 数值溢出，或 mask 设置错误 | 见 Part 2 数值稳定性 |

经验文件：[exp_debug_004](../../raw/experiences/general_debug/exp_debug_004.yaml)，[exp_debug_005](../../raw/experiences/general_debug/exp_debug_005.yaml)

---

## Part 2：数值稳定性

### 2.1 exp 溢出防护（最重要）

fp16 下 `exp(x)` 在 `x > 65504` 时溢出为 Inf。**所有含 exp 的算子必须做稳定化**：

```python
# 错误：直接 exp，fp16 易溢出
p = tl.exp(x)
s = tl.sum(p)

# 正确：减去最大值后 exp
x_fp32 = x.to(tl.float32)
m = tl.max(x_fp32, axis=0)
p = tl.exp(x_fp32 - m)
s = tl.sum(p)
# 最终结果与直接 exp 数学等价：exp(x_i) / Σ exp(x_j) = exp(x_i - m) / Σ exp(x_j - m)
```

经验文件：[exp_stability_001](../../raw/experiences/general_stability/exp_stability_001.yaml)

### 2.2 fp16/bf16 累加器溢出

归约（sum/mean/var）在 fp16 下容易累积误差，必须使用 fp32 累加器：

```python
# 错误：fp16 累加，精度损失严重
acc = tl.zeros([BLOCK], dtype=tl.float16)
acc += tl.load(ptr + offsets, mask=mask)

# 正确：fp32 累加器
acc = tl.zeros([BLOCK], dtype=tl.float32)
data = tl.load(ptr + offsets, mask=mask).to(tl.float32)
acc += data
result = acc.to(tl.float16)  # 最终再转回 fp16
```

经验文件：[exp_stability_002](../../raw/experiences/general_stability/exp_stability_002.yaml)

### 2.3 除零防护

```python
# 防止除零
eps = 1e-6
denom = tl.maximum(denominator, eps)  # 或加 eps
result = numerator / denom
```

经验文件：[exp_stability_003](../../raw/experiences/general_stability/exp_stability_003.yaml)

### 2.4 rsqrt vs sqrt+div

RMSNorm 中计算 `1/sqrt(x)` 优先使用 `tl.rsqrt`，比 `1 / tl.sqrt(x)` 精度更好且更快：

```python
# 推荐
inv_rms = tl.rsqrt(mean_sq + eps)

# 次选
inv_rms = 1.0 / tl.sqrt(mean_sq + eps)
```

经验文件：[exp_stability_004](../../raw/experiences/general_stability/exp_stability_004.yaml)

---

## Part 3：API 语法规范速查

### 禁用语法与替代方案

| 禁止 | 替代 | 经验文件 |
|------|------|---------|
| `return` / `break` / `continue` | mask 控制执行区域 | [exp_api_001](../../raw/experiences/general_api/exp_api_001.yaml) |
| `lambda` 表达式 | `tl.where` / 普通函数 | [exp_api_001](../../raw/experiences/general_api/exp_api_001.yaml) |
| `a and b and c` 链式布尔 | `mask1 & mask2 & mask3` | [exp_api_001](../../raw/experiences/general_api/exp_api_001.yaml) |
| `tensor[i]` 直接索引 | `tl.load(ptr + i)` | [exp_api_001](../../raw/experiences/general_api/exp_api_001.yaml) |
| `tl.swizzle2d` | 使用 `grouped_launch_diagonal` | [hardware/ascend_patterns](../hardware/ascend_patterns.md) |
| int64 CMP/ADD | 转 fp32 或 int32 | [exp_ascend_where_001](../../raw/experiences/general_ascend/exp_ascend_where_001.yaml) |
| 负数偏移 | `tl.maximum(offset, 0)` | [exp_api_002](../../raw/experiences/general_api/exp_api_002.yaml) |
| `tl.constexpr` host 侧 | 普通 Python 变量 | [exp_constexpr_001](../../raw/experiences/general_ascend/exp_constexpr_001.yaml) |

### API 使用注意事项

```python
# tl.where 替代条件分支
result = tl.where(condition_mask, value_if_true, value_if_false)

# mask 组合（用按位与，不用 and）
valid = (offsets < n) & (offsets >= 0)

# 负偏移保护
safe_offs = tl.maximum(offsets, 0)
data = tl.load(ptr + safe_offs, mask=(offsets >= 0) & (offsets < n), other=0.0)
```

经验文件：[exp_api_002](../../raw/experiences/general_api/exp_api_002.yaml)，[exp_api_003](../../raw/experiences/general_api/exp_api_003.yaml)，[exp_api_004](../../raw/experiences/general_api/exp_api_004.yaml)

---

## 相关页面

- [hardware/ascend_patterns.md](../hardware/ascend_patterns.md) — Ascend 特有语法约束
- [hardware/constraints_cheatsheet.md](../hardware/constraints_cheatsheet.md) — 硬件约束速查
- [workflows/profiling.md](profiling.md) — 性能分析指南
- [techniques/data_types.md](../techniques/data_types.md) — 数据类型与精度
