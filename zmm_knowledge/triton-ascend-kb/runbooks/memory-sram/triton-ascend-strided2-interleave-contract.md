---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: Triton-Ascend strided-2/interleave 访存契约（三个正确性缺陷 + 寄存器重排正解）
description: 配对类算子（RoPE interleave 等）的 stride-2 全局 load/store 在 Triton-Ascend 上有三个实测正确性缺陷（向量 div/mod×strided-2 静默错值、并发 masked strided-2 store 相邻行污染、InterleaveStatusWithMaskOptimization 丢 mask）；正解是全程连续 load/store + tl.split/tl.interleave 寄存器级重排。
tags: [strided_load, interleave, deinterleave, tl_split, tl_interleave, backend_bug, silent_corruption, masked_store, rope, correctness]
confidence: verified
status: verified
created_at: "2026-09-09T00:00:00Z"
updated_at: "2026-09-09T00:00:00Z"
---

## Triton-Ascend strided-2/interleave 访存契约（来源：ApplyRotaryPosEmb P4 三组回放，2026-09-08）

【优化维度】：访存｜正确性契约
【泛化层级】：条件性（迁移条件：任何需要奇偶/半宽配对访存的 kernel——RoPE interleave、复数运算、IQ 交织等；失效边界：后端版本修复后需复测）

<原则：配对类访存（stride=2 的 even/odd 两路半宽 load/store）在 Triton-Ascend 当前版本上**不可用**——不是性能问题而是正确性问题。一律改写为「连续整段 load → tl.reshape(…,2) + tl.split 寄存器分离 → 计算 → tl.interleave/tl.join 重组 → 连续 store」。>

### 三个实测缺陷（均在 ApplyRotaryPosEmb 45-case 基准上复现）

1. **向量 div/mod 行解码 × strided-2 访存 → 静默错值**。kernel 内用向量 `row//N`、`row%S` 解码行号、再叠加 stride-2 的 pe/po 访存时，结果错误且**无任何报错**；小 shape 单核（grid=1）时正确，规模/并发上来才错——极具迷惑性。规避：行解码退化为**标量**循环（bs 外循环），或干脆连续访存。
2. **并发下 masked strided-2 store → 非确定性相邻行污染**。同一 kernel grid=1 正常、多核并发偶发错值（NaN/垃圾），非确定复现。两组实验独立撞上（首轮 verify interleave 7/7 全挂 / bisect 定位）。
3. **`InterleaveStatusWithMaskOptimization` pass 对带 mask 的移位/重叠 load 静默丢 mask** → 越界读（507035，地址布局相关，仅在特定分配历史下触发）。规避：load 全部地址 clamp 到合法域且不带 mask，仅 store 保留 mask。

### 正解模式（实测 45/45 通过，interleave case 4.5~7.0x vs torch）

```python
# x 行连续整段 load（[BLOCK_N, 2*BLOCK_DH] 覆盖 [0, RD)）
xrow = tl.load(x_ptr + p, mask=m, other=0).to(tl.float32)
# 寄存器级 deinterleave：(…, W, 2) → even/odd
xe, xo = tl.split(tl.reshape(xrow, (BLOCK_N, BLOCK_DH, 2)))
oe = xe * ce - xo * se          # 配对数学在 even/odd 两半上做
oo = xo * co + xe * so
o = tl.interleave(oe, oo)       # 寄存器重组回连续布局（tl.join+reshape 等价）
tl.store(out_ptr + p, o.to(out_ptr.dtype.element_ty), mask=m)   # 单次连续 store
```

要点：load/store 全是普通连续 masked 流，不触发上述任何缺陷路径；`tl.split`/`tl.interleave`/`tl.join`/`tl.reshape` 在 Triton-Ascend 上可用且高效（微基准 maxdiff=0），是配对类算子的**安全路径**。

### 关联

- 应用实例：[ApplyRotaryPosEmb 语义转化卡](../../ops/apply-rotary-pos-emb.md) §7/§8（interleave 已定局为该路径）
- 移位 load 降级见 [Triton-Ascend 移位 load 降级契约](../compute-pipeline/triton-ascend-shifted-load-degradation.md)；`tl.trans` 标量化见 [tl.trans 标量化灾难与寄存器重排替代](../compute-pipeline/triton-ascend-trans-scalarization.md)

<!-- okf:related:start -->

# 相关

- 相关主题: [fp32 精度契约（长链复合算子低精度 verify 不可达的破解模式）](../algorithm-numerics/fp32-precision-contract.md) — 同属优化知识，主题相关
- 相关主题: [tl.trans 标量化灾难与寄存器重排替代（split/interleave/reshape 安全集）](../compute-pipeline/triton-ascend-trans-scalarization.md) — 主题相关：共享 tl_interleave, tl_split
- 相关主题: [scatter 语义正确性--last-write-wins + 边界检查 + 排除位置独立 kernel](../meta-skills/semantic-correctness.md) — 同属优化知识，主题相关
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](../portability/exp_rope_llama4_npu_001.md) — 主题相关：共享 rope
- 相关主题: [Triton-Ascend mask 纪律与编译期陷阱（fp32 mask 污染 / stride-0 memref / 禁 % 运算符 / 2D mask store 丢格）](../portability/triton-ascend-mask-codegen-traps.md) — 同为后端正确性缺陷契约卡（masked store 污染同症状）
- 相关主题: [device-side gather 消除 strided tile load（访存结构重组，本例 89x）](device-side-gather-strided-load.md) — 主题相关：共享 strided_load

<!-- okf:related:end -->
