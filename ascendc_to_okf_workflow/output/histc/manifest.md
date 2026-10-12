# P0 材料清单 — histc（histogram_fixed_width）

> 生成时间：2026-09-18｜来源目录：`src/ops-math/math/histogram_fixed_width/`
> **kernel 类型判定：AIV-only SIMT 直方图（scatter 累加）算子**——`KERNEL_TYPE_AIV_ONLY`（histogram_fixed_width.cpp:73），三实现类全部走 `asc_simt`（THREAD_NUM=512 线程/核，histogram_fixed_width.cpp:18）+ `asc_atomic_add`。GEIR-only（无 aclnn 层，仅 examples/test_geir_*.cpp）。**非壳算子**：全部调度/计算本体在仓（~900 行）。
>
> **🔺 低价值预判：非低价值——P2.5 降级执行①编译门+②正确性门，P4 不豁免。**
> - **❌ 项 ≥3 候选**：① `asc_atomic_add` UB 私有直方图（full_load 档，UB 可寻址 scratch）Triton 无对应；② `SetAtomicAdd<int32>+DataCopyPad` 的 GM 原子累加写回（not_full 档）；③ `SyncAll()` 清零屏障；④ SIMT 线程模型本身（512 线程 × asc_ldcg 标量 load）无对应。
> - **后端陷阱待取证**：`tl.atomic_add` 向量索引（scatter atomic）在本后端的 lowering/正确性/标量化——与 C-018（离散 gather 标量发射 ~24x）同族，**①编译门必须实测**；这是本算子第一决策点。
> - 有特殊值语义面（NaN/±inf 三组分支）、整型/浮点双 bin 计算式（int64 乘除截断 vs fp32 invRange 乘）、nbins/range 为 tensor 输入（ValueDepend，host tiling 读值）。
> - 联动结论：**走 P2.5 降级路径（①+②）**——BLOCK 选型与瓶颈定型保持两态，confidence 不升 measured(skeleton)；P4 正常执行。

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/histogram_fixed_width_def.cpp` | ✅ | x：{F32,F16,I64,I32}（:22-26）；range 同型 [2]（:27-32，**ValueDepend OPTIONAL**）；nbins：I32 标量 tensor（:33-37，ValueDepend）；**y：I32 [nbins]，InitValue(0)**（:38-42）；Attr dtype 预留默认 3（:43）；**仅 ascend950**（:49）；DynamicRank/Shape | §2、§3 |
| GEIR 原型 | `op_graph/histogram_fixed_width_proto.h` | ✅ | 登记用 | §3 |
| kernel 入口 | `op_kernel/arch35/histogram_fixed_width.cpp` | ✅ | **THREAD_NUM=512**（FPGA 128，:14-21）；`ComputeTypeOf`：f32/f16→float、i32/i64→**int64**（:31-42）；三 loadMode 分派（:44-67）；AIV_ONLY（:73） | §4.2、§6 |
| **DoBin 核心（三档共用语义）** | `op_kernel/arch35/histogram_fixed_width_simt_full_load.h:67-99` | ✅ | **特殊值**：v==+inf / NaN / (v==-inf 且 min≠-inf) → 跳过（:72-74）；**min==-inf → 全部进末 bin**（:76-77）；**max==+inf → 全部进 bin 0**（:78-79）；否则 clamp[min,max] → 整型 `(c-min)*bins/range`（int64 截断）/ 浮点 `(c-min)*invRange*bins`（fp32 截断 int32）→ idx clamp[0,bins-1]（:81-96）；`asc_atomic_add(y+idx,1)`（:98） | §6.1 |
| UB_FULL 档（loadMode=0） | 同上 :102-210 | ✅ | 每核 UB 私有直方图（bins×i32，:158/186-187）→ SIMT 线程 UB 原子加 → `SetAtomicAdd<int32>+DataCopyPad` 一次 GM 原子累加（:201-205）；前置 SimtCleanWsReduce 清 GM + **SyncAll**（:178-183）；UNROLL=6 + asc_ldcg（:110-136） | §6.2 |
| UB_NOT_FULL 档（loadMode=1） | `..._simt_not_full_load.h` | ✅ | bins 超 UB → 分 ubLoop 段，**每段重扫全量 x**、只统计落在 [ubBase,ubEnd) 的 bin（:115-117, 132-133），段内 UB 私有 + 段末 GM 原子写回（AtomicAddToGm :215-226） | §6.3 |
| GM_ATOMIC 档（loadMode=2） | `..._simt_not_full_load_simt.h` | ✅ | 无 UB buffer，**每元素直接 `asc_atomic_add` GM**（DoBinAtomic :99）——Triton 唯一可同构的官方路径 | §6.4 |
| tiling host 侧 | `op_host/arch35/histogram_fixed_width_tiling_arch35.cpp`（232 行） | ✅ | **三档决策**（:161-169）：`bins < (ubSize-32KB)/4` → UB_FULL；否则 `totalLength > bins/100`（**GM_ATOMIC_ADD_FACTOR=100**，:36 in .h）→ UB_NOT_FULL；否则 GM_ATOMIC；x 均分 formerLength=ceil(total/coreNum)（:171-173）；清 y 均分（:175-181）；**TilingInputsDataDependency({range, nbins})**（:230）；range dtype 必须==x dtype（:86-91）；min<max 校验（:108-115） | §4.1 |
| 常量 | `..._tiling_arch35.h:30-42` | ✅ | HFW_SIMT_DCACHE_SIZE=32KB（UB 可用额扣 SIMT dcache）；RANGE_LENGTH=2 | §4.1 |
| 文档 | `README.md` | ✅ | 越界归边语义（<min→首 bin，>max→末 bin）；平台表（950PR/DT、A3、A2、Atlas 推理 ✓；200I/500 A2、Atlas 训练 ✗）——**与 def 仅 950 的矛盾**：非 950 平台走其他实现（AICPU/旧仓），登记 | §2、§3 |
| golden | `tests/assets/histogram_fixed_width_spec.py` | ✅ | golden(:26-105) 与 kernel DoBin **逐式同构**（fp32 域 `diff*inv_range*bins` 同序、int64 域 `(c-min)*bins/range`、±inf 三分支一致）；customize_inputs(:107-135) 把 float x 预缩放到 [0,nbins]（测试输入生成，非算子语义） | §测试设计 |
| ST/UT | `tests/ut/op_host/arch35/test_..._tiling.cpp` + infershape UT | ✅ | host tiling + infershape 校验；无 ST atk json、无 benchmark 脚本 | §测试设计 |

## 语义速览（P1 详抽对象）

**y[b] = #{i : bin(x_i) == b}**，bin 划分 [min,max] 等宽 nbins 段，越界归边（README）。计算域：f32/f16→fp32（`(c−min)·invRange·bins` 截断），i32/i64→**int64**（`(c−min)·bins/range` 截断）。特殊值：+inf/NaN 丢弃；−inf 仅当 min=−inf 时计入（进末 bin）；min=−inf 全体进末 bin；max=+inf 全体进首 bin。

三档执行策略（tilingkey LoadMode 0/1/2，host 按 bins vs UB 与 totalLength vs bins/100 选）：
1. **UB_FULL**：核私有 UB 直方图 + UB 原子加 + 一次 GM 原子写回（最优，bins 小）；
2. **UB_NOT_FULL**：UB 分段私有直方图，**每段重扫全 x**（ubLoopNum 遍读 x）；
3. **GM_ATOMIC**：逐元素 GM 原子加（bins 大且 x 短时，省 ubLoop 遍重扫）。

## 缺省项与注意点（P1 必须处理）

1. **aclnn 层缺失**：GEIR-only——torch.histc 对应关系：torch.histc(input, bins, min, max) 返回**与输入同 dtype**（浮点），本算子 y 固定 int32；对标时语义对齐、dtype 差异登记。
2. **三档中两档 Triton 结构性 ❌**（UB 可寻址 scratch + SetAtomicAdd DataCopyPad）——Triton 仅 GM_ATOMIC 档可同构；UB_FULL 的「核私有直方图」在 Triton 的等价物不存在（无 UB 寻址），替代候选 = **向量化 bin 归属矩阵计数**（每 program 拥有 BLOCK_BINS 个 bin，扫 x tile 做等值比较累加，无原子）——P2 给两路线。
3. **`tl.atomic_add` 向量索引 scatter** 后端表现未取证（C-018 同族标量化风险）——①编译门第一实测点；若编译失败/错值，路线 2 成主路径。
4. **int64 bin 计算**（i32/i64 输入）：`(c−min)*bins` 可超 int32（大 range×bins）——kernel 必须 int64 向量算术（4a 标量化预警；但官方本就是 SIMT 逐元素标量，**标量化相对官方不是回归**，卡片写明）。
5. nbins/range 是 tensor 输入 + host tiling 值依赖——Triton wrapper 按 Python int/float 标量接（GEIR 图模式常量场景）；**运行时变 nbins 场景 host 重 tiling**，登记。
6. y 清零：def InitValue(0) + kernel 清 GM + SyncAll——Triton 侧 wrapper `torch.zeros` 即可（流内序保证，无需 SyncAll 等价物）。
7. fp16 输入 bin 计算在 **fp32 域**（COMPUTE_TYPE=float）——必须保留；int 域无 inf 语义（整型分支天然跳过 inf 检查，C++ 比较不触发）。

## P0 质量门结论

材料齐全且全部仓内一手（GEIR-only 无 aclnn 层属正常形态，非缺省）→ **通过，进入 P1**。判**非低价值**（❌≥3、scatter 原子语义、特殊值分支面、tl.atomic_add 后端未取证）→ **P2.5 降级执行①编译门+②正确性门**，P4 不豁免。
