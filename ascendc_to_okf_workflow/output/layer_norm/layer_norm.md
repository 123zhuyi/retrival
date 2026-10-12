---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: LayerNorm（沿尾部任意轴归一化，Norm 家族首卡）Ascend C → Triton 语义转化卡
description: y=(x−mean)·rstd·gamma+beta，rstd=1/sqrt(var+eps) 沿尾部 normalized_shape 折叠轴 R。官方三算法家族（Welford 单遍统计精度最优 / TwoPass 二分累加超长轴兜底 / TwoPassPerf (x−mean)复用+FusedMulDstAdd 行配对极致性能），host 按优先级 4000/200/150+IsCapable 两级筛选。半壳成分=TwoPass 的 AscendC::LayerNorm 高阶 API 与 WelfordUpdate/WelfordFinalize/Normalize 向量内置不在仓（TwoPassPerf 643 行全开实现+630 行设计文档补偿）。Triton 主路径 in-kernel two-pass：tl.sum 树形归约天然=二分累加 O(log n) 误差同构。②门抓出骨架关键 bug：非 2 幂 R 掩码尾 tile 的 var 第二遍会把掩码元素 (0−mean)²=mean² 误加进 var（高估 pad·mean²/R）→ tl.where 掩码处 d 归零。full fp32 计算域。mean/rstd 恒 f32 双输出。
tags: [layer_norm, layernorm, norm_family, welford, two_pass, binary_accumulation, rstd, newton_raphson, optional_gamma_beta, mean_rstd, mask_tail_trap, half_shell, persist_core, ascend950, ascend350, ascendc_kernelwiki]
confidence: inferred
status: draft
resource: src/ops-nn/norm/layer_norm_v4/（CANN ops-nn 本地快照，arch35 regbase 三家族 + design doc）
created_at: "2026-09-18T00:00:00Z"
updated_at: "2026-09-18T00:00:00Z"
---

sources:
  - url: src/ops-nn/norm/layer_norm_v4/op_kernel/arch35/layer_norm_v4_two_pass_perf.h
    role: primary
  - url: src/ops-nn/norm/layer_norm_v4/docs/ParallelWelford_and_BinaryAccumulation_LayerNormV4.md
    role: primary
  - url: src/ops-nn/norm/norm_common/op_kernel/reduce_common_regbase_part1.h
    role: primary

# LayerNorm（y = (x − mean) × rstd × gamma + beta）

沿尾部 `normalized_shape` 折叠成 R 轴做归一化。def 表：x∈{f32,f16,bf16}（5 组对应 gamma/beta dtype 组合），normalized_shape 为输入 tensor（i32），gamma/beta OPTIONAL（nullptr=恒 1/恒 0），**y 同 x dtype，mean/rstd 恒 f32**，Attr epsilon 默认 1e-5。对应 `torch.nn.functional.layer_norm`。regbase（910_9372）恒走 LayerNormV4；非 regbase 按 IsV4SocCheck 或回退 V3（implMode 仅 V3 消费）。**半壳**：TwoPass 用 `AscendC::LayerNorm` 高阶 API、Welford 用 `WelfordUpdate/Finalize/Normalize` 向量内置（实现不在仓）；TwoPassPerf 全开在仓（643 行），另有 630 行设计文档（金矿）。

## 1. 功能与公式（详设 §1）

- `y[m,r] = (x[m,r] − mean[m]) · rstd[m] · gamma[r] + beta[r]`；`mean=Σx/R`；`var=Σ(x−mean)²/R`；`rstd=1/sqrt(var+eps)`。
- **数值核心**：官方主算法 Welford 用在线递推 `delta=x−mean; mean+=delta/n; M2+=delta·(x−mean_new); var=M2/N`，从根源避开 `E[X²]−E[X]²` 的灾难性抵消（f16/bf16 下更显著，设计文档 §3）；超长轴（R 使 M2 累加器超 UB）退 TwoPass+二分累加。
- **二分累加**（TwoPass，设计文档 §4）：R 拆 binaryAddQuotient（从 VL_F32=64 翻倍至 ≥R 再除 2）+ remainder，各 64 子块 ReduceSum 得叶子，二叉归并树 K 层（`tmp[j]+=tmp[j+curNum]`），误差 O(n)→O(log n)（Higham §4）。**存在理由：R 超大时 Welford 的 M2 数组（每行一个）放不进 UB**。
- TwoPassPerf 加强：第一遍算 mean **同时存 (x−mean)** 到 UB（3 次读→2 次）、行配对 + `FusedMulDstAdd`（y=y·gamma+beta 单周期）、掩码位用 padded 域 `n=1/2^p` + `nCorrectionFactor=2^p/R` 修正、rstd Newton-Raphson 精炼 + inf/0 边界 Select。

## 2. 原型与约束（详设 §2）

| 项 | 值 |
|---|---|
| x | {F32,F16,F16,BF16,BF16}（由 gamma/beta dtype 组合），ND 0-8 维 |
| normalized_shape | **输入 tensor(i32) 非 attr**，0D/1D，len ≤ x 维数 |
| gamma/beta | OPTIONAL（nullptr=恒 1/0）；非 f32 须与 x 同型；shape=normalized_shape |
| y | 同 x；mean/rstd 恒 F32（shape=x 去尾部 normDim 维） |
| attr | epsilon: f32=1e-5 |
| 计算域 | **统计+归一化全 fp32**（f16/bf16 x/gamma/beta load 升 fp32，y store 降回——perf 全开实证） |
| 平台 | 910_9372=regbase arch35 三家族；910b membase 六模板裁剪（single_read/transpose/general/align/align_limit/one） |
| 确定性 | 行主语独占写（M 均分核、R 不跨核），天然确定，无 atomic |

## 3. 分核与算法（详设 §4/§5 → Triton grid）

Ascend C 事实：host 按注册优先级 **Welford(4000) > TwoPass(200) > TwoPassPerf(150)** 逐级 IsCapable——Welford（isRegBase，最宽松）→ TwoPass（+UB aFactor 二分搜）→ TwoPassPerf（+rowAlign≤16384 且 !(colSize≥4096&&rowAlign≥8192)）。外层 M 均分 coreNum 行块、R 轴核内 tile 化（UB 约束）。TILING_KEY=家族百位×权重+dtypeKey → 300/400/500 家族 × {f32ff, f16+f32w, f16ff16, bf16+f32w, bf16fbf16}。

Triton 映射——**单 kernel 行主语 persistent-core，in-kernel two-pass**：

```python
grid = (min(M, nvc),)   # program 独占一行，串行扫 R（官方 R 不跨核同构）
for m in range(pid, M, num_progs):
    # pass1: s=Σ x（fp32 域）；mean = tl.sum(s,0)/R
    # pass2: d=(xv−mean)·mask；v=Σ d²；（关键：掩码元素 d 必须归零）
    #        var=tl.sum(v,0)/R；rstd=1/tl.sqrt(var+eps)
    # pass3: yv=(xv−mean)*rstd(*gamma[+beta])；store 降回 dtype
    # mean/rstd 按 NEED_MEAN_RSTD 写 f32 到 mean_ptr/m、rstd_ptr/m
```

- **Welford/二分累加/FusedMulDstAdd/行配对/ShiftLefts = 硬件指令级优化，机制 ❌ 不逐式译**；`tl.sum` 树形归约天然=二分累加 O(log n) 精度增益（免费获得，精度准则成立）。
- **mapping 判定树**：逐元素 Welford 递推（标量链，无 SIMD 收益 + C-026 风险）❌；in-kernel two-pass ✅；块级 Chan 并行合并备选不实现。
- 三遍数据访问：官方 TwoPass 同形态；perf 的 (x−mean) UB 复用为 UB 驻留机制 ❌ 不译（Triton 大 tile 由编译器 spill 决策，两态）。

## 4. BLOCK / tile（①编译门实测）

- BLOCK_N=min(1024, next_pow2(R))；行主语 1D tile（周 dstride-free row + offs 运行传参），无 %（C-001 不命中）。BLOCK_M 不用（每 program 一行）。
- ①门附带取证：**`tl.rsqrt` 本后端编译+执行 OK**（910_9372+CANN 9.1.0，对照 1/torch.sqrt maxerr 1.19e-7 正确）——主路径用更保守的 `1/tl.sqrt`（T1 [SRC]），rsqrt 为经证备选（已回流 tl_op_lowering.md）。
- **⚠️ 掩码尾 tile trap**：pass2 方差第二遍若对掩码元素沿用 `other=0`（xv=0）→ 由 `d=xv−mean=0−mean`、`d²=mean²` 会把**假方差 pad·mean²/R** 加进 var（非 2 幂 R 才暴露，②门实测抓出 R=1000/700）。**修复：`d = tl.where(msk, xv−mean, 0.0)`**。同族思想：被掩码的「归约后置减」运算必须显式归零，勿依赖 other=0 的隐含语义。

## 5. 计算语义 → Triton（核心映射）

```python
# fp32 全计算域（perf 全开实证）；rstd=1/sqrt(var+eps)，Newton-Raphson/边界 Select 以容差吸收
for r0 in range(0, R, BLOCK_N):
    offs = r0 + tl.arange(0, BLOCK_N); msk = offs < R
    xv = tl.load(row + offs, mask=msk, other=0).to(tl.float32)
# HAS_GAMMA/HAS_BETA constexpr（nullptr=恒1/0 → 分支消除，对应 MulDstAdd 退化路径）
# mean/rstd 恒 f32 双输出；NEED_MEAN_RSTD constexpr（torch 常不取）
```
- 混合 dtype（x=f16 + gamma=f32）：Triton 指针自带 dtype 各自 `.to(fp32)`，无需 constexpr dtype 码。
- gamma/beta nullptr → HAS=0，跳过乘/加（与官方 nullptr 直传标志同构）。

## 6. 可达性总表（详设 §6）

| Ascend C | Triton | 可达性 |
|---|---|---|
| 尾部 normDim fold (M,R) | wrapper fold + contiguous | ✅ |
| M 均分核、R 核内 tile | 行主语 persistent-core + R tile | ✅ 同构（C-011） |
| Welford 单遍（WelfordUpdate 内置） | two-pass tl.sum（树形=二分累加同构精度） | ⚠️ 机制不同、精度语义同构（②门验证） |
| 二分累加 quotient/K | tl.sum 树形归约 | ✅ 精度语义同构 |
| (x−mean) UB 复用（3→2 读） | 三次 GM 读 | ❌ UB 驻留机制不译（性能两态） |
| FusedMulDstAdd/行配对/ShiftLefts | mul/add 普通指令 | ❌ 指令级优化不译（性能两态） |
| Newton-Raphson rstd + 边界 | 1/tl.sqrt(var+eps) | ✅ 语义同构（容差吸收，rsqrt ①门取证） |
| gamma/beta nullptr | HAS_*/NEED_* constexpr | ✅ |
| mean/rstd 恒 f32 双输出 | NEED_MEAN_RSTD + f32 store | ✅ |
| 空 tensor（M=0 / R=0→mean=0,rstd=NaN） | wrapper 早退 | ✅ |
| 三家族 tilingkey 分派体系 | 单 kernel 自适应 tile | ⚠️ 分派不译（shape 自适应覆盖） |
| AscendC::LayerNorm / Welford* 库 API | tl.sum/逐元素算术 | ⚠️ 半壳，机制推断段已标注 |

## 7. 精度策略与验收

- 全 fp32 计算域必须保留（f16/bf16 load 升 fp32，即使 gamma/beta 全半）。容差 f32 1e-4 / f16 1e-3 / bf16 1e-2。
- rstd=1/sqrt(var+eps)（Newton-Raphson 与直算差 <1ulp，容差吸收）；var=0 恒行 → y=beta。
- mean/rstd（恒 f32）单独比对（rtol 1e-4，本例 maxerr ~1e-6）。
- ①门覆盖：3 dtype × HAS_G/HAS_B 四组合 × R{64,1000,2048,700} × 混合精度 × 恒定行 × 大数值 1e6。

## 8. P2.5 实测摘要（①+②门，2026-09-18，910_9372+CANN 9.1.0）

- ①门：全配置真编译通过；**补证 `tl.rsqrt` 后端可用（T1，已回流 tl_op_lowering.md）**，主路径用 1/tl.sqrt。
- ②门：**16/16 PASS**；其中首轮**抓出掩码尾 tile var 高估 bug**（R=1000/700 FAIL，maxy 异常）→ `tl.where(msk, d, 0)` 修复后全绿。修复前均值/归一化都对（mean 用 other=0 正确），仅方差累积受掩码污染——值得作为 Norm/RMS 类算子的通用防坑。
- BLOCK 选型与瓶颈定型两态（降级纪律，confidence 不升 measured）。