---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: "3.3"
title: 数学归一化与算法前置条件必须显式处理（FFT 归一化/方向交替/模归约）
description: FFT 归一化、chirp 角度模归约、bitonic 方向交替、DC 虚部归零等数学约束必须显式处理，禁止依赖隐式行为
tags: [fft_normalize, chirp_modulo, bitonic_direction, dc_imag_zero, math_precondition]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-5 数学归一化前置条件

【优化维度】：数值｜数值稳定
【泛化层级】：条件性（迁移条件：算法含数学前置/后置归一化约束；失效边界：纯 element-wise 无数学约束）

<原则：FFT 归一化、chirp 模归约、bitonic 方向交替、DC 虚部归零等数学约束若遗漏会导致 7+ case 精度失败或逐 bit 不匹配，且问题隐蔽（小 shape 正常、大 shape 才暴露）。必须显式处理，禁止依赖隐式行为。>

<骨架（占位名伪码）：
# FFT 归一化：rfft 输出除以 fft_size
out = out / fft_size
# chirp 模归约：w[n]=exp(-πi·n²/N) 角度做 (n*n) % (2*N)
angle = (n * n) % (2 * N) * (-3.14159265358979 / N)
# bitonic 方向交替：左右半 DESCENDING 与 not DESCENDING 交替
# DC/Nyquist 虚部归零：实输入 RFFT 的 k=0 和 k=seqlen 虚部强制 0
out[0] = out[0].real  # 虚部归零
>

**坏实践（反例）**：依赖隐式行为或遗漏归一化
```python
out = fft(x)           # 未除以 fft_size，未匹配 torch.fft.rfft 约定
angle = n * n * (-pi/N)  # 未模归约，大 n 时 NPU cos/sin 精度退化
```
后果：7+ case 精度失败，大 shape 才暴露，问题隐蔽。✅ 正解即显式处理所有数学前置/后置约束。

**已知实例**：math-compute M6（HyenaFft FFT 归一化 L1.4、chirp 模归约 L1.6、DC 虚部归零 L3.5；Sort bitonic 方向交替 L2.5）

<!-- okf:related:start -->

# 相关

- 实践案例: [Elementwise absolute value (Abs)](../../ops/elementwise-abs.md) — 同属优化知识，主题相关
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 相关主题: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](../memory-sram/normalize-dim-to-last.md) — 同属优化知识，主题相关
- 相关主题: [非 2 的幂的语义维度必须 pad 到 next_pow2 + 极值填充 + 截断](../memory-sram/pad-next-pow2-extremum.md) — 同属优化知识，主题相关
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](../portability/exp_rope_llama4_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](two-pass-reduce-then-apply.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
