# P2 语义转化 — abs（Ascend C 事实 → Triton 决策）

> 生成时间：2026-09-17｜依据：`extraction.md`｜**P2.5 skipped: low-value**（实数域），证据 [SRC]/[DOC]。

## 0. 检索记录（backend_facts）

| 决策点 | 命中 | 出处 |
|---|---|---|
| `tl.abs` dtype | vabs 支持 **F16/F32/I8/I16/I32/I64，无 BF16**；标量化列 = i64 | dtype_support.md:143 [SRC] |
| bf16 怎么办 | regbase `isHIVMSupportedType` 不含 bf16 → 前端必须升 fp32——**与源码 AbsDag\<bf16,float\> 完全同构** | dtype_support.md:76 [SRC] |
| int64 abs | 4a'：**确认标量化**（950 无豁免）→ 性能预警，正确性无碍 | dtype_support.md:122 [SRC] |
| mask 写法 | load/store mask 保整型比较（正确性优先例外） | dtype_support §3 规则 2、C-002 |
| 对齐/寻址/双缓冲 | VV 32B（C-010）；4e int64 标量基址；multibuffer 默认 true | 同 gelu mapping §0 |

## 1. 分核策略 → grid

与 gelu 完全同构：persistent-core 连续段 `grid = min(cdiv(NUMEL, BLOCK), num_vectorcore)`，`per_prog = cdiv(NUM_BLOCKS, num_progs)`，核内连续段循环，尾块 mask。等价论证见 gelu mapping §1（同一家族 base 库 et）。

## 2. dtype 分派 → 单 kernel + 一个 constexpr 精度档

Ascend C 事实：3 tilingkey → 精度策略两档：bf16 升 fp32；其余实数原生。
Triton 决策：dtype 仍由指针推导，但**计算域不能一刀切**——按检索 §0，vabs 无 bf16：
- bf16 → `.to(tl.float32)` → abs → `.to(bf16)`（RTNE）——与源码/golden 同构，也是硬件强制；
- fp16/fp32/int → 原生 `tl.abs(x)` 不升 fp32（官方精度路径如此，fp16 升 fp32 属**偏离**而非改进，extraction §7-5）。
实现：kernel 内 `if X_DTYPE_IS_BF16:（constexpr）` 单分支，或统一写法 `x.to(promote(x.dtype)).abs()`——wrapper 按 `x.dtype == torch.bfloat16` 传 constexpr `PROMOTE_F32`。选后者（单 kernel 无分支，计算域由调用侧决定，与 tilingkey 的 host 侧分 key 同构）。

## 3. Compute → Triton 计算骨架

| Ascend C | Triton | 可达性 |
|---|---|---|
| CopyIn/CopyOut | tl.load/tl.store + mask | ✅ |
| Cast NONE / RINT | `.to()`（恒等或 RTNE） | ✅ |
| Vec::Abs | `tl.abs`（fp16/fp32/int 原生；bf16 经 fp32） | ✅（§0） |
| Reg VL 循环 | 编译器自动 | ⚠️ |
| ElementwiseSch16B | persistent-core 连续段 | ✅ 等价 |

## 4. 片上内存 → 预算

bf16+BLOCK=8192 最坏 ~96KB < 192/248KB（[SRC]）✅；其余 dtype 更低。候选 {1024,2048,4096,8192}。

## 5. 大 numel 寻址（4e 防御）

同 gelu：int64 标量基址 + int32 块内偏移 + 整型 mask。int64 输入 dtype 时指针算术不受影响（基址 int64 标量，与元素 dtype 无关）。

## 6. complex 域 ❌ → 降级方案

- ❌ 根因：Triton 无 complex 原生 dtype；complex32/64 在 aclnn 层是独立 dtype 而非两个实数张量。
- 降级方案（若必须支持）：按 golden 证实的 **packed 布局**处理——输入视为末维 2n 的实数张量，`split → re,im → tl.sqrt(re*re+im*im)`（fp32 域）→ 输出 fp32/fp16。**非正规数防御（SCALE_UP/DOWN 位操作）不翻译**，登记为精度差异风险（源为防御 denormal 下溢，Triton 直算在 |re|,|im| < 2⁻¹⁴⁹ 区间可能丢精度）。
- 本卡验收域不含 complex（ST complex case 缺席 atk json 主表）。

## 7. 精度策略 → 验收

- 分档精度路径必须保留：bf16→fp32 域；fp16/fp32/int 原生。验收对 torch.abs golden。
- int8 −128 回绕与 np 一致，无需处理。
- int64 case 验收正确性即可，性能标量化已预警（卡片 §10）。

## 8. P2 质量门自查

- [x] tl.abs lowering 依据 [SRC]；bf16 升 fp32 有源码+硬件双重佐证
- [x] ❌=1（complex）有降级方案；⚠️=2（VL、multibuffer）有出处
- [x] UB 预算校核；无 %/无 int64 lane/mask 整型
- [x] **P2.5 skipped: low-value**（实数域）
