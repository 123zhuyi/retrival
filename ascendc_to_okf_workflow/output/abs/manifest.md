# P0 材料清单 — abs

> 生成时间：2026-09-17｜来源目录：`src/ops-math/math/abs/`
> **kernel 类型判定：AIV-only 纯 elementwise**（`abs_apt.cpp:26` `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`；tiling 委托 base 库 ElewiseBaseTiling）。**壳算子**：kernel 入口 3 个 tilingkey 分支（101 bf16 / 102 实数原生 / 103 complex）→ `ElementwiseSch16B<0, AbsDag/AbscomplexDag>`；计算 DAG 在仓（`abs_dag.h` 39 行 + `abs_complex_dag.h`）；base 库 elewise tiling 在仓（`model/opbase-master-src-op_common/.../elewise_tiling.cpp`）。
>
> **🔻 低价值预判：低价值（实数域）——P2.5 skipped: low-value；P4 降单组冒烟（实数 dtype）。**
> - 实数域 ❌=0：`Vec::Abs` 一条原语，落在 mapping §3 ✅ 行（`tl.abs`）；无广播/规约/gather；公式 y=|x| 在基座先验内；无 benchmark 语义超集（ST 仅 dtype/shape/值域变化，golden=torch.abs）。
> - **complex 域例外**：complex32/complex64 记 1 个 ❌（Triton 无 complex 原生 dtype）——属扩展 dtype 族，不影响实数域低价值判定；❌ 项按纪律在 mapping 给出降级方案，卡片 §9 登记。验收域 = 实数 dtype。
> - 联动结论：① 本行即「P2.5 skipped: low-value」记录；② 卡片 confidence=inferred；③ 证据上限 [DOC]/[SRC]；④ 不产 p25_report.md；P4 降单组冒烟。

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/abs_def.cpp` | ✅ | x REQUIRED 9 型 {BF16,F16,F32,INT8,INT16,INT32,INT64,COMPLEX32,COMPLEX64}（:23-30）；y 同型，complex32→F16、complex64→F32（:31-38）；DynamicRank/Shape=true（:42-43）；PrecisionReduceFlag(true)（:45）；**AICore 注册 ascend950 + ascend350 双平台**（:47-48） | §2、§3 |
| GEIR 原型 | `op_graph/abs_proto.h` | ✅ | 登记用 | §3 |
| shape 推导 | `op_host/abs_infershape.cpp` | ✅ | elewise 直通（输出=输入 shape；complex 时输出 dtype 变但 shape 不变） | §2.1 |
| tiling host 侧 | `op_host/arch35/abs_tiling_arch35.cpp` | ✅ | 3 个 tilingkey 常量（:13-15）：101=BF16、103=COMPLEX、102=其他实数；**分 key 逻辑按 outputDtype==BF16 / inputDtype==COMPLEX**（:24-31）；dtype 校验：非 complex 时 x/y 必须同型，complex64→FLOAT、complex32→FLOAT16（CalcOutputDtype :36-70）；workspace=0（:17,23）；委托 ElewiseBaseTiling | §4.1.3 |
| kernel 入口（壳） | `op_kernel/abs_apt.cpp`（73 行） | ✅ | AIV-only（:26）；tilingdata=`EleBaseTilingData16B`（:27-28）；101→`AbsDag<bfloat16_t, float>`（:31）；102→6 路 constexpr：half→**AbsDag<half,half>（fp16 原生计算不升 fp32）**、float→float、int8/16/32/64 原生（:34-59）；103→complex64→`AbscomplexDag<complex64,float>`、complex32→`<complex32,half>`（:60-69） | §4.1、§6 |
| **计算 DAG（核心语义）** | `op_kernel/arch35/abs_dag.h`（39 行） | ✅ | `AbsDag<U,T=float>`：CopyIn→Cast<T,U,0>→**Vec::Abs\<T\>**→Cast<U,T,1>→CopyOut（:28-32）；MemCfg LEVEL_2（:35）。T 的实例化策略见 kernel 壳：bf16→T=float（升精度）、其余→T=U（原生） | §1、§6、§7 |
| complex DAG | `op_kernel/arch35/abs_complex_dag.h` | ✅ | AbscomplexCustom Reg 链：模长 sqrt(a²+b²)，含**非正规数防御**（SCALE_UP=2²⁴ 放大→fp32 计算→SCALE_DOWN 缩回、指数偏置 0x3F800000 常数） | §6（❌ 域） |
| base 库 tiling | `model/.../atvoss/elewise/elewise_tiling.cpp` | ✅（在仓） | 同 gelu/add：cache-line 对齐连续段 + ubFormer 核内循环 | §4.1 |
| aclnn 接口 | `op_api/aclnn_abs.{h,cpp}`、`abs.{h,cpp}` | ✅ | 两段式直通 | §2.2 |
| 文档 | `README.md`；`docs/aclnnAbs.md` | ✅ | y=\|x\| | §1.1 |
| golden | `tests/assets/golden.py` | ✅ | 实数：np.abs（**bf16 显式升 fp32 再回原型**——与 DAG 一致）；complex32：`np.split(x,2,axis=-1)` 拆实虚部 → `sqrt(re²+im²)`（**packed 布局：末维=2n，前半实部后半虚部**）；aclnn golden=torch.abs | §测试设计 |
| ST 用例 | `tests/st/aclnnAbs/atk_aclnnAbs.json`（200 case） | ✅ | dtype 分布：fp32×40、fp64/int32/uint8/bf16/fp16/int16/int64/int8 各×20；值域 nd(mean[-100,100],std[1,25])；acc=high_precision。**⚠️ fp64/uint8 不在 def dtype 表内——ST 跨平台超集，以 def 为准** | §测试设计 |
| UT | `tests/ut/op_host/arch35/test_abs_tiling_arch35.cpp` 等 | ✅ | host/api 层 | §测试设计 |

## 计算 DAG 语义速览（abs_dag.h）

实数域单原语：`Cast<T,U,NONE> → Abs<T> → Cast<U,T,RINT>`。**精度策略按 dtype 分三档（kernel 壳决定 T）**：

| 输入 U | 计算 T | 语义 |
|---|---|---|
| bf16 | **float** | 升 fp32 取绝对值，RTNE 回写（golden 同构） |
| fp16 | **half** | fp16 原生（不升 fp32）——与 gelu 的恒 fp32 策略不同 |
| fp32 | float | 原生 |
| int8/16/32/64 | 原生 | 整数绝对值 |
| complex64/32 | float/half | 模长 sqrt(re²+im²) + 非正规数防御（❌ 域，见 manifest 头注） |

## 缺省项与注意点（P1 必须处理）

1. **调度器 ElementwiseSch16B 本体不在仓**——同 gelu/add 先例，base 库 et 锚定 §4，执行骨架 inferred。
2. **ST dtype 超集**：fp64/uint8 在 ST 200 case 中出现但 def 不支持——以 def 为准，卡片 §9 标偏差。
3. **fp16 原生 vs bf16 升 fp32 的非对称精度策略**是本算子唯一的 dtype 陷阱点，P2 需对照 backend_facts 取证 vabs 的 dtype 支持。
4. **int64 abs**：源生支持，但 Triton 侧 int64 向量算术标量化（dtype_support 4a 已知），P2 登记性能预警。
5. **complex packed 布局**：golden 显示末维 split 实虚部（非交错），若 Triton 要支持需按 packed 语义拆——降级方案方向。
6. 双平台注册（950+350）：卡片 §3 记 350 亦在支持列。

## P0 质量门结论

材料齐全（proto/infershape/tiling/kernel/DAG×2/docs/tests/golden + base 库 tiling 在仓）→ **通过，进入 P1**。低价值（实数域）判定成立：**P2.5 skipped: low-value**，P4 降单组冒烟，confidence=inferred。
