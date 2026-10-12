# P1 语义抽取 — abs

> 生成时间：2026-09-17｜依据：`output/abs/manifest.md`（低价值·实数域，P2.5 skipped: low-value）
> 锚点格式：`file:line`，根目录 `src/ops-math/math/abs/`；et = base 库 elewise_tiling.cpp（在仓）。

## 1. 功能与公式

- `y = |x|` 逐元素（README；docs/aclnnAbs.md）。实数域：绝对值；complex 域：模长 `sqrt(re²+im²)`。
- 无广播、无规约；输出 shape = 输入 shape（elewise 直通 infershape）。
- golden：实数 `np.abs`（**bf16 显式升 fp32 计算再回原型**，golden.py）；complex32 packed 布局 `np.split(x,2,-1)` → `sqrt(re²+im²)`；aclnn golden = `torch.abs`。

## 2. 接口与原型

### 2.1 GEIR op（abs_def.cpp）
- `x` REQUIRED：9 型 {BF16,F16,F32,INT8,INT16,INT32,INT64,COMPLEX32,COMPLEX64}（:23-30），ND。
- `y`：非 complex 时与 x **同型**（tiling 层强校验，abs_tiling_arch35.cpp:40-47）；complex32→**F16**、complex64→**F32**（:48-70，def.cpp:31-38）。
- DynamicRank/Shape=true（:42-43）；PrecisionReduceFlag(true)（:45）；**AICore 双平台 ascend950 + ascend350**（:47-48）。

### 2.2 aclnn 层（op_api/aclnn_abs.cpp）
- 两段式直通，无属性，workspace=0（abs_tiling_arch35.cpp:17,23）。

### 2.3 原型约束
- 0~8 维全动态（DynamicRank）；无近似/模式开关。

## 3. 规格约束

| 项 | 值 | 锚点 |
|---|---|---|
| dtype（实数） | bf16/fp16/fp32/int8/int16/int32/int64，in/out 同型 | def.cpp:25；tiling:40-47 |
| dtype（complex） | complex32→fp16、complex64→fp32 | def.cpp:33；tiling:48-70 |
| shape | 0~8 维全动态 | def.cpp:42-43 |
| 平台 | ascend950 + ascend350 | def.cpp:47-48 |
| ST 值域 | nd(mean[-100,100], std[1,25]) | atk_aclnnAbs.json |

**偏差**：ST 200 case 含 fp64/uint8（各 20）——不在 def dtype 表，ST 跨平台超集，以 def 为准。

## 4. 分核策略与 tiling（✅ base 库源码锚点）

### 4.1 分核
- AIV-only（abs_apt.cpp:26）；tilingdata = `EleBaseTilingData16B`（:27-28）。
- 分核机制同 gelu/add 家族：展平一维、核数折算 capped by AIV 核数、cache-line 对齐连续段 blockFormer + 尾核、核内 ubFormer 循环（et）。

### 4.2 tilingkey 语义（本算子的 dtype 策略载体）
| key | 触发条件 | kernel 实例 | 计算域 |
|---|---|---|---|
| 101 | outputDtype==BF16 | `AbsDag<bfloat16_t, float>` | **升 fp32** |
| 102 | 其他实数 | half→`AbsDag<half,half>`；float→float；int8/16/32/64 原生 | **原生（fp16 不升！）** |
| 103 | inputDtype∈{COMPLEX32,COMPLEX64} | `AbscomplexDag<complex64,float>` / `<complex32,half>` | 模长+非正规数防御 |

（abs_tiling_arch35.cpp:24-31 分 key；abs_apt.cpp:30-70 分派。）

### 4.3 BLOCK 候选
- persistent-core 连续段范式同 gelu（mapping §1 [EXP]）；BLOCK 候选 {1024..8192}，VV 32B 对齐全满足。

## 5. 片上内存

- MemCfg LEVEL_2（UB，abs_dag.h:35）；bf16 路径多一份 fp32 中间；无 workspace。BLOCK=8192 bf16 最坏 ~96KB < 192/248KB ✅。

## 6. 流水编排（计算 DAG —— 核心语义资产）

### 6.1 实数 DAG（abs_dag.h:26-37）
```
AbsDag<U, T> = CopyIn<U> → Cast<T,U,0(NONE)> → Vec::Abs<T> → Cast<U,T,1(RINT)> → CopyOut<U>
```
- **T 由 kernel 壳按 dtype 决定**（§4.2）：bf16→float（升精度），其余实数→T=U（原生，两次 Cast 恒等）。
- 单条 Vec::Abs 原语——本算子是比 gelu 更纯粹的「Cast+单指令+Cast」骨架。

### 6.2 complex DAG（abs_complex_dag.h，❌ 域）
- 模长 sqrt(a²+b²) Reg 链，含**非正规数防御**：SCALE_UP=2²⁴ 放大 → fp32 计算 → SCALE_DOWN=2⁻²⁴ 缩回；指数偏置常数 0x3F800000；half 路径用 uint16 位操作。
- golden 证实 packed 布局（末维 split 实虚部）。Triton 无 complex 原生 dtype → ❌，降级方案见 mapping。

## 7. 数值精度

- **非对称精度策略（本算子唯一 dtype 陷阱）**：bf16 升 fp32 + RTNE 回写；fp16 原生 fp16；整型原生。golden 与之一一对应（bf16 显式升 fp32）。
- 硬件根源（P2 取证）：vabs 指令支持 F16/F32/I8/I16/I32/I64、**无 BF16**（dtype_support.md:143 [SRC]）——源码策略即硬件 dtype 支持的直译。
- int64：源生支持；Triton 侧 int64 vabs 标量化（dtype_support:143 标量化列 [SRC]）——性能预警非正确性问题。
- int8 边界：|−128| 溢出回绕（−128）——np.abs(int8) 同行为，golden 一致，无语义差。

## 8. 测试与标杆

- golden：§1；ST：200 case、9 dtype（含超集 fp64/uint8）、值域 nd(mean[-100,100],std[1,25])、acc=high_precision。
- UT：host 层 tiling/infershape。
- 无仓内 benchmark；验收标杆 = torch.abs，实数 dtype。

## 9. 偏差与风险清单

1. ST fp64/uint8 超集 ≠ def（§3）——以 def 为准。
2. complex 域 ❌（Triton 无 complex dtype）——降级方案在 mapping §6；验收域不含 complex。
3. 调度器本体不在仓——同 gelu 先例，inferred 不影响转化。
4. int64 Triton 标量化——性能预警，若 benchmark 含 int64 case 需在卡片明示。
5. fp16 原生计算（不升 fp32）——生成 Agent 若套用 gelu「恒升 fp32」习惯会**偏离官方精度路径**（结果可能反而更准但与 golden 回写路径不同），卡片必须明示分档。

## P1 质量门

- [x] 三层语义齐备（kernel DAG / aclnn / 测试层）
- [x] 关键论断带 file:line 锚点；§4 分核为 base 库源码事实
- [x] 实数域 ❌=0；complex ❌ 已识别并规划降级
- [x] 低价值（实数域）复核通过 → P2，**P2.5 skipped: low-value**
