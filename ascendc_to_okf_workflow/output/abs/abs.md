---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: Abs（实数绝对值 + complex 模长）Ascend C → Triton 语义转化卡
description: 最纯粹的「Cast+单指令+Cast」elementwise 算子 y=|x|。AIV-only，kernel 3 个 tilingkey 分支（bf16 升 fp32 / 实数原生 / complex 模长）。核心语义资产 = 非对称精度策略：bf16 升 fp32、fp16 原生不升——backend_facts 取证证实这就是 vabs 指令 dtype 支持（F16/F32/I8/I16/I32/I64、无 BF16）的直译，生成 Agent 照 gelu 习惯「恒升 fp32」会偏离官方路径。complex 域（模长 sqrt(re²+im²) + 非正规数防御，packed 布局）对 Triton 是 ❌（无 complex dtype），降级方案=末维 split 实虚部 fp32 域直算。P0 判低价值（实数域），P2.5 skipped: low-value，P4 降单组冒烟。
tags: [abs, elementwise, aiv_only, bf16_promote, fp16_native, int64_scalarize, complex_fallback, ascend950, ascend350, ascendc_kernelwiki]
confidence: inferred
status: draft
resource: src/ops-math/math/abs/（CANN ops-math 本地快照，ascend950 + ascend350）
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-09-17T00:00:00Z"
---

sources:
  - url: src/ops-math/math/abs/
    role: primary
  - url: model/opbase-master-src-op_common/src/op_common/atvoss/elewise/elewise_tiling.cpp
    role: base_library

# Abs（y = |x|）

逐元素绝对值（实数域）；complex 域为模长 `sqrt(re²+im²)`。**AIV-only**（abs_apt.cpp:26）。壳算子：3 个 tilingkey 分派（abs_tiling_arch35.cpp:24-31 → abs_apt.cpp:30-70），计算 DAG 在仓（abs_dag.h 39 行；abs_complex_dag.h）。分核/BLOCK 机制有 base 库 elewise_tiling.cpp 源码锚点（同 gelu/add 家族）。

## 1. 功能与公式（详设 §1）

- 实数：`y = |x|`；complex：`y = sqrt(re² + im²)`（complex32→fp16 输出、complex64→fp32 输出）。
- 输出 shape = 输入 shape；无属性、无 broadcast/规约。
- golden：实数 np.abs（**bf16 显式升 fp32 再回原型**）；complex packed 布局 split 后 sqrt(re²+im²)（golden.py）。

## 2. 原型与约束（详设 §2）

### 输入输出（GEIR，abs_def.cpp:23-38）
- `x` REQUIRED 9 型：{BF16,F16,F32,INT8,INT16,INT32,INT64,COMPLEX32,COMPLEX64}，ND。
- `y`：实数与 x 同型（tiling 强校验，abs_tiling_arch35.cpp:40-47）；complex32→F16、complex64→F32。
- DynamicRank/Shape=true（0~8 维）；**AICore 双平台 ascend950 + ascend350**（:47-48）。
- ⚠️ ST 200 case 含 fp64/uint8 超集（atk json）——不在 def 表，以 def 为准。

### aclnn 层
两段式直通；无属性；workspace=0（tiling:17,23）。

## 3. 规格约束（详设 §3）

| 项 | 值 |
|---|---|
| dtype | 实数 7 型同型进出；complex32→fp16、complex64→fp32 |
| shape | 0~8 维全动态，ND |
| 平台 | ascend950、ascend350 |
| ST 值域 | nd(mean[-100,100], std[1,25])，acc=high_precision |

## 4. 分核策略（详设 §4 → Triton grid）

与 gelu/add 家族同一 base 库机制：展平一维、核数折算、cache-line 对齐**连续段** + 核内 ubFormer 循环（elewise_tiling.cpp 在仓）。
Triton 映射 = persistent-core 连续段：`grid = min(cdiv(NUMEL, BLOCK), num_vectorcore)`（运行时查询禁写死）；`per_prog = cdiv(NUM_BLOCKS, num_progs)` 连续段循环；尾块 mask。等价论证见 gelu 卡 §4。

## 5. BLOCK / Tiling 参数（详设 §4.1.3）

- tilingkey 三键（101 bf16 / 102 实数 / 103 complex）→ Triton 侧由**指针 dtype + 一个 constexpr `PROMOTE_F32`** 吸收（wrapper 判 `x.dtype == torch.bfloat16` 传入，对应 host 侧分 key）。
- BLOCK 候选 {1024, 2048, 4096, 8192}，key=["NUMEL"]；VV 32B 对齐全满足（C-010）。

## 6. 片上内存预算（详设 §5）

最坏 bf16+BLOCK=8192 ≈ 96KB < UB 192KB(A2)/248KB(950) [SRC] ✅；实数原生路径更低。MemCfg LEVEL_2；无 workspace。

## 7. 流水编排与 Triton 可达性（详设 §6）

| Ascend C | Triton | 可达性 |
|---|---|---|
| CopyIn/CopyOut | tl.load/tl.store + mask | ✅ |
| Cast NONE/RINT | `.to()`（恒等或 RTNE） | ✅ |
| Vec::Abs（实数） | `tl.abs` | ✅ vabs F16/F32/I8/I16/I32/I64 [SRC] |
| Vec::Abs（bf16） | `.to(fp32)→abs→.to(bf16)` | ✅ 硬件强制路径（vabs 无 BF16 [SRC]） |
| int64 abs | `tl.abs`（int64） | ⚠️ **标量化**（dtype_support:143 [SRC]）——正确性无碍、性能预警 |
| AbscomplexDag（模长+denormal 防御） | 无 complex dtype | ❌ 降级：packed split → fp32 `sqrt(re²+im²)`；denormal 防御不翻译（§9-2） |
| Reg VL 循环 / DAG 数据流同步 | 无对应 | ⚠️ 编译器自动 |
| ElementwiseSch16B 调度 | persistent-core 连续段 | ✅ 等价 |

## 8. Triton 骨架（生成用伪码）

完整骨架 `output/abs/triton_skeleton.py`（preflight 0 命中）。核心：

```python
x = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
if PROMOTE_F32:                       # bf16：源码 AbsDag<bf16,float> 同构
    y = tl.abs(x.to(tl.float32)).to(x_ptr.dtype.element_ty)
else:                                 # fp16/fp32/int：原生，不升 fp32
    y = tl.abs(x)
tl.store(y_ptr + base + cols, y, mask=mask)
```

## 9. 精度标杆与验收（详设 §7/§8）

- golden = torch.abs（实数）；**bf16 标杆是升 fp32 路径**（golden.py 显式 float()），Triton 同构。
- **必须保留**：分档精度路径——bf16 升 fp32；fp16/fp32/int 原生。**禁止对 fp16 也升 fp32**（偏离官方回写路径，extraction §7-5）。
- int8 |−128| 回绕与 np 一致，无需特判。
- ST 域：9 dtype（含超集 fp64/uint8，以 def 7 实数型为准）× 值域 nd(mean[-100,100],std[1,25])。
- P4 策略（低价值联动）：单组冒烟——bf16/fp16/fp32/int32 各一档对 golden；验收标准「无负引导」。

## 10. Agent actions（生成指引）

1. 识别类别：纯 elementwise 一元，AIV-only，无 dot/规约/广播；单输入单输出。
2. 分核：persistent-core 连续段（§4 范式），grid 核数运行时查询；尾块 mask。
3. **必须保留的语义点**：① bf16→fp32 计算域（硬件强制，vabs 无 BF16）；② 其余 dtype 原生不升；③ in/out 同型。
4. **不要复刻**：3 路 tilingkey 壳（指针 dtype + PROMOTE_F32 吸收）、aclnn 直通层、schMode。
5. dtype 不做 kernel 内分支：一个 constexpr PROMOTE_F32 由 wrapper 决定（对应 host 分 key 的位置）。
6. int64 输入可用但预警：vabs-i64 标量化 [SRC]，大 shape int64 case 性能差属预期。
7. complex 不在本卡验收域；若被要求支持，走 §7 降级（packed split），勿尝试位操作 denormal 防御的逐位复刻。
8. 无属性、无 % 解码、无 num_stages/num_warps；禁凭空引入 approximate/eps 之类参数。

## 11. 复现性声明

本卡由 Ascend C 源码静态分析转化（**confidence: inferred**；P2.5 skipped: low-value·实数域，证据上限 [SRC]/[DOC]）。锚点指向 `src/ops-math/math/abs/`，分核事实补充自 op_common base 库 elewise_tiling.cpp；后端论断经 backend_facts 检索取证（vabs dtype 矩阵、i64 标量化、mask 规范、C-010）。未经实测的点：BLOCK autotune 优选、int64 标量化的实际劣化幅度、bf16 升 fp32 与原生的性能差。残留不确定项：ElementwiseSch16B 本体不在仓（inferred，不影响语义）；complex denormal 防御未翻译（精度风险已登记）；ST fp64/uint8 超集来源未追溯（他平台实现）。回放验证：P4 降单组冒烟，报告见 `output/abs/validation_report.md`。
