---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: Gelu（tanh 近似，sigmoid 形态实现）Ascend C → Triton 语义转化卡
description: 纯 elementwise 激活算子，aclnn/GEIR 语义一致 y = GELU(x, approximate="tanh")。AIV-only，kernel 入口是 3 路 dtype 分派壳（gelu_apt.cpp → ElementwiseSch16B），核心语义资产是仓内计算 DAG（gelu_dag.h）：全程 fp32 中间域（Cast NONE 进 / RINT 出），Reg 链用 Axpy 常数融合把 (x+0.044715x³) 一项算成 x³+x/0.044715。分核/BLOCK 机制有 op_common base 库 elewise_tiling.cpp 源码锚点（cache-line 对齐连续段均分 + 核内 ubFormer 循环）。Triton 转化天然单 kernel：dtype 由指针推导、无属性、无 workspace；后端检索确认 vexp/vdiv 仅 F16/F32（fp32 域计算是硬性约束而非仅精度策略）。P0 判低价值（❌=0、无广播/规约/gather、公式在基座先验内、无 benchmark 超集），P2.5 skipped: low-value，P4 降单组冒烟。
tags: [gelu, activation, elementwise, aiv_only, tanh_approx, sigmoid_form, fp32, fp16, bf16, ascend950, ascendc_kernelwiki]
confidence: inferred
status: draft
resource: src/ops-nn/activation/gelu/（CANN ops-nn 本地快照，ascend950 only）
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-09-17T00:00:00Z"
---

sources:
  - url: src/ops-nn/activation/gelu/
    role: primary
  - url: model/opbase-master-src-op_common/src/op_common/atvoss/elewise/elewise_tiling.cpp
    role: base_library

# Gelu（tanh 近似）

GEIR/aclnn Gelu：逐元素 `y = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`（README；docs/aclnnGelu.md）。**kernel 实际用 sigmoid 形态** `y = x / (1 + e^(-1.5957691(x + 0.044715x³)))`（gelu_dag.h:29-30 注释自述）——数学恒等（0.5x(1+tanh(z)) = x·sigmoid(2z)，2z = √(8/π)(·) ≈ 1.5957691(·)）。**AIV-only**（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，gelu_apt.cpp:31；TilingPrepare 取 `GetCoreNumAiv()`，gelu_tiling_arch35.cpp:144）。**壳算子**：kernel 入口 3 路 dtype constexpr 分派（gelu_apt.cpp:33-43）→ `ElementwiseSch16B<schMode, GeluDAG<U>::OpDag>`；调度器本体不在仓，但 base 库 tiling 在仓（elewise_tiling.cpp）→ §4/§5 有源码锚点。golden 即 `torch.nn.GELU(approximate="tanh")`（tests/assets/golden.py:30-45）。

## 1. 功能与公式（详设 §1）

- 语义：`out = GELU(x, approximate="tanh")`，**本仓无 approximate 属性、无 erf 形实现**——tanh 形是唯一语义。
- 逐元素、无广播、无规约；输出 shape = 输入 shape（委托 base 库 InferShape4Elewise，gelu_infershape.cpp）。
- sigmoid 形全值域数值一致性（extraction §7 已论证）：大正 x → e^t 下溢 0 → y=x ✓；大负 x → e^t 上溢 +inf → y→−0，与 tanh 形极限 0 一致 ✓；ST 含 fp16 inf 值域 case 作官方背书。

## 2. 原型与约束（详设 §2）

### 输入输出（GEIR，gelu_def.cpp:23-32；gelu_proto.h 一致）

- `x` REQUIRED：{BF16, F16, F32}，ND；`y` 同型同 format。**无属性**。
- DynamicRank/DynamicShape=true（def.cpp:37-38）：0~8 维全动态（README）。
- ⚠️ 文档偏差：docs/aclnnGelu.md 产品支持表含 310p/A3/A2 等，但 def.cpp:42 仅注册 ascend950——**以 def 为准**（旧平台实现在他仓）。

### aclnn 层（aclnn_gelu.cpp）

- 两段式接口；dtype 白名单 {FLOAT, FLOAT16, BF16}（:23-24）；**bf16 平台门槛 DAV_2201 或 Regbase**（:28）。
- 非连续输入 `Contiguous` 化（:112）→ `l0op::Gelu`（:116）→ 非连续输出 `ViewCopy` 回写（:120）。空 tensor 早退。
- workspace = 0（gelu_tiling_arch35.cpp:22,118）——单 kernel 直通，无 GM 中转。

## 3. 规格约束（详设 §3）

| 项 | 值 | 来源 |
|---|---|---|
| dtype | fp16 / bf16 / fp32（in/out 同型） | gelu_def.cpp:25,30 |
| shape | 0~8 维，全动态 | README；def.cpp:37-38 |
| format | ND | def.cpp:26 |
| 平台 | ascend950（本仓 def）；aclnn bf16 需 DAV_2201/Regbase | def.cpp:42；aclnn_gelu.cpp:28 |
| ST 值域 | [-7,7] 为主，含 fp16 inf 边界 case | atk_aclnnGelu.json |

## 4. 分核策略（详设 §4.1 → Triton grid）

源码事实（elewise_tiling.cpp 在仓）：展平一维 → 核数 = min(按 MIN_TILING_BITS_SIZE_PER_CORE 折算, AIV 物理核数) → 每核 **cache-line 对齐连续段** `blockFormer = ceil(ceil(dim0/coreNum)/CACHE_LINE)·CACHE_LINE` + 尾核 blockTail → 核内 ubFormer=maxElemNum 循环（ubLoopOfFormerBlock/ubLoopOfTailBlock/ubTail*）。
**Triton 映射（忠实等价）**：persistent-core 连续段范式——

```python
grid = (min(triton.cdiv(numel, BLOCK), NUM_VEC_CORE),)   # 核数运行时查询 num_vectorcore，禁写死
# kernel 内：
per_prog = tl.cdiv(NUM_BLOCKS, num_progs)
for blk in range(pid * per_prog, min((pid + 1) * per_prog, NUM_BLOCKS)):  # 连续段，禁交织
```

尾块 mask 统一处理，不拆 normal/tail 分支。等价论证：核间连续段 + 核内 UB 块两级切分 = 连续段循环每 iteration 一个 BLOCK，负载等价（mapping §1 [EXP 2026-09-07]）。

## 5. BLOCK / Tiling 参数（详设 §4.1.3）

### tilingkey → 无需映射
`GET_TPL_TILING_KEY(1, dType)`（gelu_tiling_arch35.cpp:120）：第一维固定 1；dtype 维被 Triton 指针类型推导天然吸收；schMode∈{0,1} 是 base 库调度内部键，Triton 无对应。**kernel 无任何 constexpr 分支、无属性**。

### BLOCK
- autotune 候选 {1024, 2048, 4096, 8192}，key=["NUMEL"]。
- 对齐：VV 类尾轴 32B 整除（C-010 [DOC]）——四候选对 fp32/fp16/bf16 全满足。
- Reg 链本身按向量寄存器宽度 VL 循环（gelu_dag.h:37-39）——BLOCK 只是 UB 块粒度，VL 细分由编译器完成，BLOCK 无需对齐 VL。

## 6. 片上内存预算（详设 §5）

BLOCK=8192 最坏情形：输入 fp16 16KB + fp32 中间峰值 ~64KB + 输出 16KB ≈ 96KB < UB 192KB(A2)/248KB(950)（hardware_platforms [SRC]）✅ 全候选安全，无需分平台降档。DAG MemCfg=LEVEL_2（UB 级，gelu_dag.h:84）；无 workspace。

## 7. 流水编排与 Triton 可达性（详设 §6）

### 可达性判定表（❌=0）

| Ascend C | Triton | 可达性 |
|---|---|---|
| LoadAlign/StoreAlign（UB 对齐搬运） | tl.load/tl.store + mask | ✅ |
| Mul/Muls/Axpy/Adds（向量算术） | `*`、标量广播、fma | ✅ vmul f32 |
| Exp | tl.exp（**必须 fp32 域**：vexp 仅 F16/F32 [SRC dtype_support:157]） | ✅ |
| Div | `x / t`（vdiv f32 [SRC]） | ✅ |
| Cast NONE 进 / RINT 出 | `.to(tl.float32)` / `.to(out_dtype)`（RTNE 默认） | ✅ 必须保留 |
| Reg VL 循环 + UpdateMask | 无对应 | ⚠️ 编译器 VL 细分，自动 |
| TQue/EnQue/DeQue（DAG Bind 数据流） | 无对应 | ⚠️ 编译器自动插同步 |
| 3 路 dtype 分派壳 | 单 kernel 指针推导 | ✅ 简化 |
| ElementwiseSch16B 调度器 | persistent-core 连续段 | ✅ 等价（§4） |
| double buffer（base 库调度内） | multibuffer 默认 true [SRC Options.td] | ⚠️ 不显式编码 |

## 8. Triton 骨架（生成用伪码）

完整可编译骨架见 `output/gelu/triton_skeleton.py`（已过语法检查与 preflight 扫描）。核心计算体：

```python
x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
x3 = x * x * x
# 常数照抄源码（gelu_dag.h:25-26），与官方舍入路径逐位同构
t = (x3 + x * (1.0 / 0.044715)) * (-1.595769121 * 0.044715)
y = x / (1.0 + tl.exp(t))
tl.store(y_ptr + base + cols, y.to(y_ptr.dtype.element_ty), mask=mask)
```

大 numel 防御（4e）：`base = blk.to(tl.int64) * BLOCK` 标量基址 + int32 块内偏移；mask 整型比较（`cols < rem`，rem 先夹回 int32）——load/store mask 保整型是官方规范（dtype_support §3 规则 2，正确性优先于 vcmp 标量化代价）。

## 9. 精度标杆与验收（详设 §7/§8）

- golden：`torch.nn.GELU(approximate="tanh")`，fp16/bf16 升 fp32 计算后回原 dtype（golden.py:36-44）——与 kernel DAG 精度路径同构。
- **必须保留**：fp32 中间域 + RTNE 回写；常数表达式照抄（消灭常数来源差异）。
- ST 域（atk_aclnnGelu.json）：fp16/fp32/bf16 × 小 shape × 值域 [-7,7]/8/inf；acc=default。
- 容差：仓内未给数值阈值 → 通用 dtype 默认容差；精度异常排查旋钮 `TRITON_DEFAULT_FP_FUSION=0`（mul+add→fma 融合开关 [DOC]）。
- P4 策略（低价值联动）：降为单组冒烟——fp32/fp16/bf16 各一档 shape 对 golden 比对即验收；预期结论「卡片增益≈0」，验收标准是**无负引导**。

## 10. Agent actions（生成指引）

1. 识别类别：纯 elementwise 一元激活，AIV-only，无 tl.dot、无规约、无广播；单输入单输出。
2. 分核：展平 1-D + persistent-core 连续段循环，grid=min(cdiv(numel,BLOCK), vector_core_num)，核数运行时查询禁写死；尾块 mask。
3. **必须保留的语义点**：① fp32 中间域（load 后 `.to(tl.float32)`，store 前 `.to(out_dtype)`）——这不只是精度策略，vexp/vdiv 向量指令仅 F16/F32 [SRC]，bf16 域直接算 exp 会标量化/不可达；② sigmoid 形态公式与常数（照抄 `1/0.044715`、`-1.595769121*0.044715`）；③ tanh 形是唯一语义，勿加 approximate 分支。
4. **不要复刻**：3 路 dtype 分派壳（指针推导天然覆盖）、aclnn 层 Contiguous/ViewCopy（约定调用方 `.contiguous()`）、schMode/tilingkey 分派。
5. dtype 不做 constexpr 分支：统一 fp32 路径覆盖三型（13→1 实例简化，add 先例）。
6. BLOCK 初选偏大导致核占不满时优先降 BLOCK（对齐 base 库核数折算语义）；autotune {1024..8192} 自动覆盖。
7. 无 % 解码、无运行时标量属性、无 num_stages/num_warps——本算子无任何属性，禁凭空引入。
8. numel ≥ 2³¹ 防御：int64 仅标量基址（§8），勿写 `offs = blk*BLOCK + tl.arange(...)` 纯 int32 形式。

## 11. 复现性声明

本卡由 Ascend C 源码静态分析转化（**confidence: inferred**；P0 判低价值 → P2.5 skipped: low-value，未经骨架实测，证据上限 [SRC]/[DOC]）。锚点指向 `src/ops-nn/activation/gelu/`，分核/BLOCK 事实补充自 op_common base 库（`model/opbase-master-src-op_common/src/op_common/atvoss/elewise/elewise_tiling.cpp`）。后端论断经 backend_facts 检索取证（vexp/vdiv dtype 约束、mask 规范、multibuffer 默认值、C-001/C-002/C-010）。未经硬件实测的点：BLOCK 最优值（autotune 候选未验证）、连续段 vs 交织在本算子的实际差（沿用 fused_bias_leaky_relu 家族 [EXP] 结论）、sigmoid 形 inf 支路的硬件实测行为（已做数值论证 + ST inf case 背书）。残留不确定项：ElementwiseSch16B 调度器本体不在仓（执行骨架 inferred，不影响语义）；MIN_TILING_BITS_SIZE_PER_CORE/CACHE_LINE_BYTE_LENGTH 常量值未从 .h 头文件确认（不影响 Triton 转化）。回放验证：未做（P4 降为单组冒烟待执行，报告将补至 `output/gelu/validation_report.md`）。
