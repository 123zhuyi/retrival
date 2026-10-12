# P1 语义抽取 — layer_norm（layer_norm_v4）

> 生成时间：2026-09-18｜依据：manifest.md 归集材料｜判定：非低价值（P2.5 降级①+②）
> 目标平台 910_9372 = arch35 regbase（isRegBase=true）→ **arch35 三家族（TWO_PASS 3xx / WELFORD 4xx / TWO_PASS_PERF 5xx）为主抽取对象**；membase(910b) 六模板与 common 兜底裁剪登记。

## 1. 算子语义（数学定义）

```
对 x 的尾部 len(normalized_shape) 个轴（折叠为 R）做归一化：
  mean[m]  = (1/R) Σ_r x[m,r]
  var[m]   = (1/R) Σ_r (x[m,r] − mean[m])²
  rstd[m]  = 1 / sqrt(var[m] + eps)
  y[m,r]   = (x[m,r] − mean[m]) · rstd[m] · gamma[r] + beta[r]
mean/rstd 输出恒 f32，shape = x.shape 去掉尾部 normDim 维（外层 M 维）。
gamma/beta OPTIONAL：nullptr 语义 = 恒 1 / 恒 0（regbase 直传 nullptr 标志位，不填常量张量——aclnn:380-414）。
eps 为 f32 attr，def 默认 1e-5；kernel 成员默认同（two_pass.h:337）。
```

## 2. 原型与约束

| 项 | 值（def.cpp + tiling_base + aclnn） |
|---|---|
| x | {F32, F16, F16, BF16, BF16} 五组（对应 gamma/beta dtype 组合），ND 0-8 维 |
| normalized_shape | **输入 tensor（i32）非 attr**，0D/1D，len ≤ x 维数（tiling_base:139-152） |
| gamma/beta | OPTIONAL；dtype ∈ {f32, f16, bf16}；**非 f32 时必须与 x 同型**（aclnn:131-141）；shape = normalized_shape |
| y | 同 x dtype 与 shape |
| mean/rstd | **恒 F32**，shape = x 去尾部 normDim 维；arch3510 下 aclnn 校验尾部维可为 1（CheckMeanRstdOutputShape） |
| attr | epsilon: float = 1e-5 |
| 平台 | def: 910b/910_93/310p + ascend950 等 config json；**910_9372 走 regbase 路径** |
| dtypeKey | tensorKey×10+paramKey（f32=0/f16=1/bf16=2，tiling_base:33-54）→ 组合：00=f32/ff、11=f16/ff16、12→f16+f32w 查表在 apt.h 模板分派 |

## 3. aclnn 接口层语义（aclnn_layer_norm.cpp）

- dtype 支持表：910={f16,f32}、910B+（含 regbase）={f16,f32,bf16}（:53-57）。
- **regbase 恒走 LayerNormV4**（`IsRegbase()` true 直返，:276-279）；非 regbase 按 IsV4SocCheck 选 V4 否则回退 V3（implMode 仅 V3 消费，V4 路径不读 implMode）。
- **空 tensor**：M=0（外层空）→ 直返（workspace=0）；**N=0（归一化轴空）→ mean=0、rstd=NaN**（ProcessEmptyTensorWithValue，:360-374）——y 空。
- 非连续输入 Contiguous 化；weight/bias 缺失时 **regbase 传 nullptr**，非 regbase Fill 1/0（:380-414）。
- 输出三件套各自 Cast+ViewCopy（mean/rstd f32→用户请求 dtype）。
- aclnn_fast_layer_norm.cpp 为另一入口（未详读，登记）。

## 4. 分核与 tiling（host 侧，全部在仓）

### 4.1 框架两级筛选（设计文档 §5）

```
注册优先级：Welford 4000 > TwoPass 200 > TwoPassPerf 150（common 2000 但与 regbase 互斥）
框架按优先级尝试，模板 IsCapable() 判定：
  Welford    : isRegBase（最宽松）
  TwoPass    : isRegBase + UB 放得下（二分搜 aFactor）
  TwoPassPerf: isRegBase + rowAlign≤16384 + !(colSize≥4096 且 rowAlign≥8192) + UB（最严格）
TILING_KEY = 家族百位 × LN_TEMPLATE_KEY_WEIGHT + dtypeKey
  → 300/310/311/320/322 = TWO_PASS × {f32ff, f16+f32w, f16ff16, bf16+f32w, bf16fbf16}
  → 400/410/... = WELFORD；500/... = TWO_PASS_PERF
```

### 4.2 公共折叠（tiling_base:160-168）

`colSize=∏x.shape[:dim−normDim]`（外层 M）、`rowSize=∏normalized_shape`（R）、`rowAlign`=R 按 32B/dtype 对齐、`vlFp32`=vectorLength/4。

### 4.3 各家族 tiling 事实

- **Welford**（welford_tiling.cpp）：M 均分 coreNum（blockFactor=ceil(M/coreNum)，:38-44）；DoLibApiTiling 从 UB 预算循环反推最大 tileLength（步进 64、双缓冲 ×2，buffer 账目：x/y/gamma/beta 双缓冲 + mean/rstd 双缓冲聚合 256 + welfordTemp + apiTemp，:49-115）；workspace 恒 16MB。
- **TwoPass**（regbase_two_pass_tiling.cpp）：**binaryAddQuotient=从 vlFp32(64) 翻倍至 ≥R 再除 2**（:63-67）；powerOfTwoForR=≥R 最小 2 幂（:131-134）；**binaryAddK=⌈log2(vcaddNum/64)⌉ 归并层数**（vcaddNum=quotient/64，≤64 时 K=0 直 ReduceSum，:148-162）；UB 二分搜 aFactor（:77-86）。
- **TwoPassPerf**：额外预算 (x−mean) 差值 buffer；阈值常量 LN_ROW_THRESHOLD=4096、LN_COL_THRESHOLD=8192（:22-23）。
- **common（非 regbase 兜底）**：tilingKey=700+dtypeKey%10；UB 128K 常量、tileLength=UB/4/2（common_tiling:69-79）——裁剪登记。

## 5. kernel 计算结构（arch35 三家族）

### 5.1 TWO_PASS（two_pass.h，358 行，半壳）

- 每核 aBlockFactor=ceil(M/blockNum) 行，UB 内 aFactor 行循环；mean/rstd GM 偏移 = aBlockFactor×blockIdx（:98-99）。
- 计算核心 = **`AscendC::LayerNorm` 高阶 API**（本体不在仓）+ 外层 `CalculateMeanVarVF` 二分累加包装。
- **二分累加**（设计文档 §4）：R 拆 binaryAddQuotient（2 幂部分）+ remainder；各 VL_F32=64 子块 ReduceSum 得叶子，tmpBuf 二叉归并 K 层（`tmp[j] += tmp[j+curNum]`），误差 O(log n)。**存在理由：R 超大时 Welford 的 M2 累加器数组（每行一个）超 UB**。

### 5.2 WELFORD（welford.h，398 行，半壳）

- Process = WelfordInitialize → ProcessWelfordUpdate（tileLength tile 循环 + tail）→ WelfordFinalize（M2/N→var→rstd）→ ProcessNormalize，**统计 1 遍 + 归一化 1 遍 = 2 次数据访问**。
- `WelfordUpdate` 向量内置：64 f32/SIMD lane 并行递推 `delta=x−mean; mean+=delta/n; M2+=delta·(x−mean_new)`（实现不在仓）；var=M2/N 无 E[X²]−E[X]² 抵消，f16/bf16 下精度优势更显著（设计文档 §3）。

### 5.3 TWO_PASS_PERF（two_pass_perf.h，643 行，全开——P2 主参照）

- **全 fp32 计算域**：LoadRegForDtype 把 f16/bf16 x 与 gamma/beta 全 Cast 到 RegTensor<float>，统计、归一化全在 fp32，**仅 store y 时降回 T**（StoreRegForDtype）。gamma/beta f16 全半配置下同样在 fp32 域乘加。
- R≤64：单 VL 掩码直算（CalculateMeanVarRLessThanVL）；R≤128：双 VL（:231+）；更大：RCommon<1|2> 二分累加，**mean 用 padded 域**：`n=1/powerOfTwoForR`，`Muls(meanSum, x, n)` 后 Reduce，再乘 **nCorrectionFactor=powerOfTwoForR/R** 修正掩码位（:194-216）——掩码位贡献 0 的优雅处理。
- 第一遍算 mean **同时存 (x−mean) 到 xSubMeanUb**（:137-156，省一次 GM 搬移：3 次→2 次）；第二遍对差值平方、按 VL 块 ReduceSum 落 tmpUb、再末级归并（LAST_LOOP_NUMS=1 直 ReduceSum / =2 ShiftLefts 指数对齐+Add，:460-475）。
- **rstd 后处理**：`NormCommon::ComputeRstdNewtonRaphson<false>`（perf :161-163）→ **rstd = Newton-Raphson 精炼 rsqrt(var+eps)**（norm_common/reduce_common_regbase_part1.h:286+）：`var+=eps; r=1/var; y=sqrt(r); t1=1.5+(−0.5·var·y)·y; rstd=y·t1` + 二次精炼（:195-211），**边界 Select：var==+inf→rstd=0；var==0→rstd=+inf**（:212-215）。NEED_MAX 变体另有 Maxs(var,−99.99) 钳制（perf 用 false 版不带）。
- **CalculateNormalizeVF**（:500+）：**行配对两行**处理，`MulDstAdd(y, gamma, beta)` 融合指令完成 y=y·gamma+beta（hasGamma&&hasBeta 时），无 gamma/beta 时退化为 Mul/Add 或直乘 rstd。
- mean/rstd 经 DataCopyPad 按 currentANum×4B 写 GM（CopyOutMean/CopyOutRstd）。

### 5.4 库 API 缺省登记

`AscendC::LayerNorm`、`WelfordUpdate/WelfordFinalize/Normalize`、`Get*MaxMinTmpSize` 实现不在仓——语义由 two_pass_perf 全开实现 + torch/golden 真值三方互证，卡片标「机制推断」。

## 6. 计算语义 → 精度域事实（P2 直接消费）

1. **统计（mean/var）与归一化全程 fp32 域**（perf 全开实现实证；f16/bf16 输入 load 即升 fp32，y store 降回）。
2. **rstd=rsqrt(var+eps)**，Newton-Raphson 精炼 + inf/0 边界 Select（语义=1/sqrt(var+eps)，精度优于朴素除法开方；Triton 侧 `tl.rsqrt` 或 1/tl.sqrt 语义同构，容差吸收）。
3. **掩码位 0 贡献 + 修正因子**：mean=Σ(x·(1/2^p))·(2^p/R)——Triton `tl.sum(masked, other=0)/R` 同构（树形归约天然 O(log n)，等价二分累加精度收益）。
4. gamma/beta nullptr → 跳过乘/加（wrapper constexpr 分支）。
5. mean/rstd 恒 f32 输出，与 y dtype 无关。

## 7. 特殊/边界语义

- M=0 → 空输出早退；N=0 → y 空、mean=0、rstd=NaN（aclnn 层语义，wrapper 复刻）。
- var=0（恒定行）→ rstd=1/sqrt(eps)，y=beta（数学）；kernel Newton-Raphson 边界 var==0→rstd=+inf 是 eps=0 极端，正常 eps>0 不触发。
- 非 32B 对齐 R：rowAlign 补齐 + 掩码，语义不受影响。
- implMode 仅 V3 路径消费，V4/regbase 忽略——本卡不映射。

## 8. 测试资产

- golden/ST/UT 在 `tests/`（登记）；P2.5 ②门基准用 torch 语义参照（`torch.nn.functional.layer_norm` fp32 参照 + 手算 mean/var），容差：f32 rtol/atol 1e-4~1e-5 级、f16 1e-3、bf16 1e-2（对齐 sum 卡纪律），mean/rstd 单独比对。

## 9. 风险清单（P2/P2.5 关注点）

1. `tl.rsqrt` 950 后端 lowering 未取证 → ①门验证（降级：1/tl.sqrt）。
2. Welford 并行递推不可直译 → P2 判定树倾向 in-kernel two-pass（tl.sum 树形归约=二分累加同构精度），块级 Chan 合并作备选。
3. 半壳库 API（LayerNorm/Welford*）内部精度细节不可见 → 以 perf 全开实现为准+torch 真值兜底。
4. R 轴大时单 program 行全载的寄存器压力 → BLOCK_N tile 化，选型两态。
5. mean/rstd 可选输出（torch 常不取）→ wrapper flag 控制，constexpr NEED_MEAN_RSTD。
6. gamma/beta 混合 dtype（x=f16+gamma=f32）→ kernel 内统一升 fp32（已实证），wrapper 直接传原 dtype 指针 + constexpr 分支。
7. f16/bf16 大 R 时精度：two-pass fp32 域足够；验收容差按 §8。
