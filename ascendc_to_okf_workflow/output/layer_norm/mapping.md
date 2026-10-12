# P2 语义转化 — layer_norm（Ascend C 事实 → Triton 决策）

> 生成时间：2026-09-18｜依据：`extraction.md`｜**非低价值 → P2.5 降级执行①编译门+②正确性门**（③③.5④ 跳过，BLOCK 选型/瓶颈定型两态）。

## 0. 检索记录（backend_facts）

| 决策点 | 命中 | 出处 |
|---|---|---|
| `tl.sum` lowering | `linalg.reduce` T2 [SRC] | tl_op_lowering.md:39 |
| `sqrt`/`precise_div` lowering | arith/math 直译 **T1** | tl_op_lowering.md:53 |
| `tl.rsqrt` | **未取证**（硬件 vrsqrt 仅 F16/F32 存在，dtype_support:160；Triton 算子映射无记录）→ ①门附带验证，主路径用 `1/tl.sqrt`（T1 已取证） | — |
| `%` 禁用 / C-001 | 行解码用 `//` + `a-(a//b)*b` | backend_contracts.md C-001 |
| stride/shape 运行传参 | 不 constexpr 化（C-017）；dtype/flag 分支 constexpr 化（C-023） | C-017/C-023 |
| **C-026 标量行循环+整型累加非确定错值** | 本算子统计域 fp32（非 i32），但结构上仍采 **2D tile (BLOCK_M, BLOCK_N) 块状加载**（sum 卡修复形态同构），不用「标量行循环+逐迭代换 base 行 load」 | C-026 |
| scatter atomic | 不涉及（行主语独占写 mean/rstd，无竞争） | C-024 不命中 |
| persistent-core 范式 | 小块海量任务 → persistent-core grid=min(tasks, nvc) | C-011 |

## 1. 分核策略 → grid

Ascend C 事实：外层 M 均分 coreNum（welford_tiling:38-44），R 轴核内 tile 化（UB 约束）；跨核不拆 R（三家族均无 R 跨核归约——与 sum 的 groupR>1 不同，layer_norm 天然行主语）。

Triton 决策——**单 kernel、program 独占 (BLOCK_M 行) 行块**，persistent-core：

- grid = min(cdiv(M, BLOCK_M), nvc)；program 内 task 循环 `for t in range(pid, NUM_TASKS, num_progs)`，t 直接×BLOCK_M 得行基（**无 % 解码**，C-001 不命中）。
- 每 task：行块 `rows = t*BLOCK_M + arange(0,BLOCK_M)`，mask rows<M；R 轴 tile 循环 `for r0 in range(0, R, BLOCK_N)`，2D load (BLOCK_M, BLOCK_N)。
- **R 轴单 program 全载**（官方同：R 不跨核）→ 确定性天然保证，无 atomic。

## 2. 统计算法选择（P2 判定树）

| 候选 | 判定 | 理由 |
|---|---|---|
| ① 逐元素 Welford 递推（mean/M2 标量链） | ❌ 不采用 | Triton 向量语义下标量递推串行化，无 SIMD 收益；C-026 同族风险 |
| ② **in-kernel two-pass**（Σx→mean；Σ(x−mean)²→var→rstd；归一化第三遍） | ✅ **主路径** | 结构=官方 TwoPass 家族同构；`tl.sum` 树形归约天然具备二分累加 O(log n) 误差特性（设计文档 §4 的精度收益免费获得）；fp32 域统计 |
| ③ 块级 Chan 并行合并（每 tile (n,mean,M2) 合并） | 备选不实现 | 数学同构 Welford 但复杂度高，②的精度经 ②门容差验证已够（f16/bf16 输入升 fp32 域后 E[X²] 抵消风险已被 two-pass 结构消除） |
| ④ E[X²]−E[X]² 单遍 | ❌ 禁用 | 灾难性抵消（设计文档 §3 核心警告） |

官方 Welford/二分累加/ShiftLefts/FusedMulDstAdd/行配对为硬件指令级优化，**机制层 ❌ 不逐式译**（精度语义由 ② 同构覆盖，性能差异两态登记）。

## 3. 计算域与 dtype → kernel 结构

- **全 fp32 计算域**（two_pass_perf 全开实证：f16/bf16 x 与 gamma/beta load 即升 fp32，y store 降回）→ kernel 内 `tl.load(...).to(tl.float32)`，store `.to(y_ptr.dtype.element_ty)`。
- gamma/beta dtype 独立于 x（混合精度组：f16+f32w 等）→ 各自指针 + 各自 `.to(tl.float32)`，无需 constexpr dtype 码（Triton 指针自带 dtype）。
- **HAS_GAMMA / HAS_BETA constexpr**（C-023）：nullptr 语义=恒 1/恒 0，分支消除乘法/加法（对应官方 MulDstAdd 退化路径）。
- **NEED_MEAN_RSTD constexpr**：不需要时不写 mean/rstd（torch layer_norm 常不取）；写时恒 f32。
- eps 运行传参（f32 标量）。

## 4. rstd → Triton

- 主路径：`rstd = 1.0 / tl.sqrt(var + eps)`（sqrt T1 [SRC] 已取证）。
- 官方 Newton-Raphson 精炼 + inf/0 边界 Select = 精度/边界机制，语义=1/sqrt(var+eps)，容差吸收（var+eps>0 时 Newton-Raphson 与直算差 <1ulp 级，两态登记）；var==0→+inf 边界仅 eps=0 触发，验收不覆盖（登记）。
- ①门附带试 `tl.rsqrt` 编译（若过，注释备选；不过则不使用）。

## 5. wrapper 归一化

```
layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, need_mean_rstd=False)
```
- normalized_shape：int 或 tuple；校验 = x 尾部 len 维逐维相等（aclnn:218-230 同构）。
- x contiguous 化；M=∏x.shape[:-ndim]，R=∏normalized_shape。
- weight/bias：contiguous+flatten (R,)；None → HAS=0。
- **M==0** → 空 y/mean/rstd 早退；**R==0** → y 空、mean=zeros、rstd=full(NaN)（aclnn 空 tensor 语义同构，:360-374）。
- 非连续输入：torch 层 contiguous（官方 aclnn 同路径，登记）。

## 6. BLOCK / 预算

- BLOCK_N = min(1024, next_pow2(R))；BLOCK_M = 8（2D tile ≤ 8×1024×4B=32KB 逻辑面，fp32 acc 同形 ×2 + diff 复算无驻留——普通 UB 预算，C-005，两态）。
- R>BLOCK_N 时 tile 循环，x 三次 GM 读（官方 TwoPass 同形态；官方 perf 的 (x−mean) UB 复用是 UB 容量机制，Triton 侧大 tile 由编译器 spill 决策，两态登记）。
- 无 dot/scan/gather/atomic——preflight 高危项天然不命中。

## 7. 可达性总表

| Ascend C | Triton | 可达性 |
|---|---|---|
| 尾部 normDim 折叠 (M,R) | wrapper fold + contiguous | ✅ 同构 |
| M 均分 coreNum、R 核内 tile | persistent-core 行块 + R tile 循环 | ✅ 同构（C-011） |
| Welford 单遍统计（WelfordUpdate 内置） | two-pass tl.sum（树形=二分累加同构精度） | ⚠️ 机制不同、精度语义同构（②门验证） |
| 二分累加 binaryAddQuotient/K | tl.sum 树形归约 | ✅ 精度语义同构（结构不逐式译） |
| (x−mean) UB 复用（3 次读→2 次） | 三次 GM 读 | ❌ UB 驻留机制不译（性能两态） |
| FusedMulDstAdd / 行配对 / ShiftLefts | mul/add 普通指令 | ❌ 指令级优化不译（性能两态） |
| Newton-Raphson rstd + 边界 Select | 1/tl.sqrt(var+eps) | ✅ 语义同构（容差吸收，①门取证 rsqrt 备选） |
| gamma/beta nullptr 标志 | HAS_GAMMA/HAS_BETA constexpr | ✅ 同构 |
| mean/rstd 恒 f32 双输出 | NEED_MEAN_RSTD constexpr + f32 store | ✅ 同构 |
| 空 tensor（M=0/N=0 NaN） | wrapper 早退 | ✅ 同构 |
| 三家族 tilingkey 分派体系 | 单 kernel 自适应 tile | ⚠️ 分派体系不译（shape 自适应覆盖） |
| AscendC::LayerNorm/Welford* 库 API | tl.sum/逐元素算术 | ⚠️ 半壳，机制推断段已标注 |

## 8. 精度策略 → 验收（②门）

- 基准：fp64 参照 `mean/var` 手算 + torch F.layer_norm 双对照；y/mean/rstd 三输出全比对。
- 容差：f32 rtol/atol 1e-4；f16 1e-3；bf16 1e-2（对齐 sum 卡纪律）；mean/rstd（f32）1e-5 级。
- 覆盖：3 dtype × {有/无 gamma × 有/无 beta} × {R 小(≤64)/R 中(1024)/R 大(>4096 非 2 幂)} × {M=1/M 大} × {need_mean_rstd 两态} × {normalized_shape 多维尾部} × {空 M}。
- 特别 case：**恒定行（var=0）**、**大数值行（检验 two-pass 抗抵消）**、混合 dtype（x=f16, gamma=f32）。

## 9. P2 质量门自查

- [x] tl.sum T2/sqrt T1 [SRC]；tl.rsqrt 未取证 → 主路径不用，①门附带验证
- [x] ❌ 项（UB 复用/融合指令/分派体系）均有降级论证；Welford→two-pass 精度同构论证（树形归约=二分累加）
- [x] C-026 规避：2D tile 块状加载；无 %；stride 运行传参；HAS_*/NEED_* constexpr
- [x] **P2.5 降级①+② 计划**：①编译门 = 3 dtype × HAS_GAMMA/HAS_BETA 四组合 × R 三档真编译（+tl.rsqrt 附带试编译）；②正确性门 = `__main__` smoke（§8 覆盖面）
