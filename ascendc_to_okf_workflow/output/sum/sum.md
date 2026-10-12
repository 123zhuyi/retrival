---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: Sum（任意轴组合规约求和，reduce 家族首卡）Ascend C → Triton 语义转化卡
description: y=Σx over axes，任意多轴组合。官方统一抽象=去1维/合轴后 AR 交错 pattern（A=规约轴 R=保留轴，1~9 维），A 轴优先分核，groupR>1 时 GM workspace 两阶段跨核归约+ScheduleMode=1 核间同步——Triton 结构性 ❌，降级为单 kernel A 轴全载（确定性反而对齐 README 声明）。DAG=Cast→ReduceSumOp→Cast：f16/bf16 升 fp32 中间域必须保留；bool 经 int64 域、y=int64。半壳算子（调度器 ReduceSch/规约原语不在仓，tiling 计算器 1293 行在仓）。P2.5 ①门抓获 C-026：标量行循环「逐迭代换 base 的行 load+i32 向量累加」非确定错值（单 program 复现）→ 2D tile 块状加载替代；i64 vreduce 本栈编译/正确性确认（A2/A3 标量化、性能两态）。
tags: [reduce_sum, sum, reduce, ar_pattern, cross_core_reduce, workspace, deterministic, promote_dtype, bool_int64, row_loop_trap, half_shell, ascend950, ascend350, ascendc_kernelwiki]
confidence: inferred
status: draft
resource: src/ops-math/math/reduce_sum/（CANN ops-math 本地快照，950+350 AICore，AICPU 兜底另册）
created_at: "2026-09-18T00:00:00Z"
updated_at: "2026-09-18T00:00:00Z"
---

sources:
  - url: src/ops-math/math/reduce_sum/
    role: primary
  - url: model/opbase-master-src-op_common/src/op_common/atvoss/reduce/reduce_tiling.cpp
    role: primary

# Sum（y = Σ x over axes，任意轴组合求和）

对 axes 任意组合维度求和（README）。属性 **keep_dims**（默认 false）、**noop_with_empty_axes**（默认 true：空 axes 直返 x）。对应 `torch.sum(input, dim, keepdim, dtype=)`，aclnn 层语义完全对齐。def：x∈{f16,f32,bf16,i32,i64,bool}，**y=bool→int64 其余同 x**（def.cpp:19-25）；950+350 双 AICore（:53-54）；**默认确定性实现**（README 约束）。**半壳**：调度器 `ReduceSch`/`Vec::ReduceSumOp`/`GetPromoteType` 定义不在仓，tiling 计算器在仓——机制推断段已标注。

## 1. 功能与公式（详设 §1）

- `y = Σ_{i∈axes} x_i`；负轴归一（+ndim 后排序，MakeWrapDim 同构）。
- 空 axes：noop=True（默认）→ identity；False → 全轴规约。axes=None（aclnn 空 dims）→ 全轴（aclnn:252-260）。
- 计算域（DAG，reduce_sum_dag.h:28-38）：`CopyIn<T> → Cast<PromoteT> → ReduceSumOp<PromoteT> → Cast<T> → CopyOut`——**先升域、规约、降回**。f16/bf16→fp32（golden 佐证）；bool→int64 域（BoolDag：i8→half→float→int64 三级链，:40-52）。
- golden（golden.py）：f16/bf16 升 fp32 求和回原型；其余 np.sum 直算（⚠️ np/torch 对 i32 默认升 i64 域，官方 kernel 的 i32 promote 缺省未证——大 R 轴 i32 溢出风险登记）。

## 2. 原型与约束（详设 §2/§3）

| 项 | 值 |
|---|---|
| x | {F16,F32,BF16,I32,I64,BOOL}，ND，支持非连续，0-8 维 |
| axes | {I32,I64} tensor 常量（ValueDepend）→ wrapper 按 Python int/tuple/None 接 |
| y | BOOL→I64，其余同 x；显式 dtype 时本卡 **store 端 cast**（⚠️ aclnn 是「先 cast 输入再规约」，中间域不同，对齐 GEIR kernel 层，差异登记） |
| 属性 | keep_dims=false；noop_with_empty_axes=true |
| 平台 | 950+350 AICore；README 的 DOUBLE/COMPLEX=仓外实现 |
| 确定性 | 默认确定性——**禁用任何 atomic 归约写法**（C-024 错值+乱序双杀） |

## 3. 分核策略（详设 §4 → Triton grid）

Ascend C 事实：任意 axes → EliminateOne/MergeAxis → **AR 交错 pattern**（A/R 奇偶位，1~9 维，>4 维 pad）；5 步 tiling（basicBlock 按 DAG buffer 需求+DB×2+cast ratio 折算 → cacheLine 切分轴 → UB 内 A 轴（≤512B 或分核<85% 调整）→ UB 内 R 轴 → 回调）；**A 轴优先分核**（numBlocks 对齐 outerA 倍数），R 轴次之；**groupR=ceil(outerR/factorRCntPerCore)>1 → ScheduleMode=1 核间同步 + GM workspace=vectorCoreNum×align(outSize×maxInputBytes, cacheLine) 两阶段归约**（tiling:1101-1145, 1249-1259）。

Triton 映射——**单 kernel A 轴全载（groupR=1 形态）**，每 program 独占输出元素、扫整条规约轴：

- **路线 T（尾轴 (M,R)→(M,)）**：persistent-core，`acc += tile` 分块循环 + `tl.sum`。
- **路线 M（中间轴 (M,R,N)→(M,N)）**：program 独占 (m, n-tile)（task 解码用 `//`+`a-(a//b)*b`，无 %），**R 分块 2D tile (BLOCK_R=8, BLOCK_N=128) 块状 load 累加**，末 `tl.sum(acc, axis=0)`。
- **wrapper 归一化**：axes 连续段 → 折 (M,R,N)；**非连续段（如 4D {0,2}）→ permute+contiguous 后走路线 T**（官方 NDDMA 免拷路径 ❌ 性能机制不译）。
- **跨核两阶段 ❌ 不译**：A 全载天然确定（对齐 README）；R 轴极大时核间并行度损失两态，「wrapper 两段式 partial+final（GM 中转=workspace 同构）」为后续优化方向，本骨架不实现。

## 4. BLOCK / tile（①编译门实测）

- 路线 T：BLOCK_R=256（fp32 域 1KB 级）；路线 M：(BLOCK_R=8, BLOCK_N=128)（i32 4KB 级）——普通 UB 预算（C-005），无 CallOp。
- **⚠️ C-026（①门抓获，已回流）**：路线 M 初版「标量 r 循环 + 每迭代换 base 的行 load」在 **i32/行宽 128（512B 行）非确定错值**——同输入两跑结果不同，**单 program 复现**；f16/i64 同构未触发；no-op cast 不解。改 2D tile 块状加载后 23/23 正确。**循环内避免「标量迭代变量×行宽」的逐行 load**。
- i64 规约（vreduce i64）本栈编译+正确性通过（②门 i64 case 全过）——与「A2/A3 标量化」[SRC] 一致：性能标量化、正确性不受限，性能两态不外推。

## 5. 计算语义 → Triton（核心映射）

```python
# ACC constexpr 码（C-023 wrapper 传入）：0=fp32 域 / 1=i32 域 / 2=i64 域
if ACC == 0:   # f16/bf16/f32 → fp32 累加（DAG Cast→ReduceSumOp→Cast 同构）
    tv = tl.load(ptrs, mask=lmask, other=0).to(tl.float32)
elif ACC == 1: # i32 直算（官方 promote 缺省；大 R 轴溢出风险登记）
    tv = tl.load(ptrs, mask=lmask, other=0)
else:          # i64/bool(int8 化) → int64 域（BoolDag 同构）
    tv = tl.load(ptrs, mask=lmask, other=0).to(tl.int64)
acc += tv                       # 2D tile 累加（路线 M）/ 1D 分块（路线 T）
s = tl.sum(acc, axis=...)       # linalg.reduce T2 [SRC]
tl.store(y_ptr + ..., s.to(y_ptr.dtype.element_ty), mask=...)
```
- bool：wrapper `.to(torch.int8)` 后 ACC=2，y=int64（def 表；aclnn 2²⁴ FLOAT 限流是接口层另一策略，不对齐，登记）。
- 空规约轴（R=0）→ wrapper 直返 zeros；空 axes → identity 早退；均不进 kernel。

## 6. 可达性总表（详设 §6）

| Ascend C | Triton | 可达性 |
|---|---|---|
| 任意 axes → AR 交错 pattern | wrapper 连续段折叠 / 非连续段 permute+contiguous | ✅ 语义同构 |
| A 轴优先分核 | 输出元素级 persistent-core | ✅ 同构（C-011） |
| **groupR>1 跨核两阶段 + ScheduleMode=1** | 单 kernel A 全载 | ❌ 核间同步结构性不可译；A 全载降级（确定性对齐）；两段式为后续方向 |
| Cast→ReduceSumOp→Cast（fp32 中间域） | load→fp32→tl.sum→store cast | ✅ 逐式同构 |
| BoolDag 三级 cast int64 域 | wrapper int8 + kernel int64 | ✅ 语义同构 |
| useNddma 非连续搬运 | permute+contiguous 拷贝 | ❌ 性能机制不译（语义等价） |
| basicBlock/resultBlock/DB 流水 | 固定小 tile | ⚠️ 机制推断（半壳）；UB 远离上限 |
| 空 tensor / 空 axes noop | wrapper 早退 | ✅ |

## 7. 精度策略与验收

- f16/bf16：fp32 中间域必须保留；容差 f16 1e-3 / bf16 1e-2（对齐 ST）。
- f32：tl.sum 树形 vs torch 顺序，R=4096 实测累加序噪声 ≤4.6e-5——rtol 1e-4 + atol 1e-4。
- i32/i64/bool：精确相等（i32 溢出场景除外，登记）。
- 验收覆盖：6 dtype × {尾轴/中间轴/相邻多轴/非相邻多轴/负轴/全轴/空 axes 两态/keep_dims/大 R 轴/1D}。

## 8. P2.5 实测摘要（①+②门，2026-09-18，910_9372+CANN 9.1.0）

- ①门抓获 1 条新后端事实并回流：**C-026**（标量行循环 i32 累加非确定错值 → 2D tile 替代，preflight #17）；确认 i64 vreduce 编译/正确性（与既有 [SRC] 一致，不新增条目）。
- ②门 **23/23 PASS**（覆盖 §7 验收面，对 golden 同构基准）。
- BLOCK 选型与瓶颈定型两态（降级纪律，confidence 不升 measured）。
