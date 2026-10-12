# P0 材料清单 — sum（reduce_sum）

> 生成时间：2026-09-18｜来源目录：`src/ops-math/math/reduce_sum/`；base 库 tiling：`model/opbase-master-src-op_common/src/op_common/atvoss/reduce/reduce_tiling.cpp`
> **kernel 类型判定：AIV-only 规约算子（半壳）**——`reduce_sum_apt.cpp` 是 bool/非 bool DAG 分派壳：计算 DAG（`arch35/reduce_sum_dag.h`）在仓，**调度器本体 `atvoss/reduce/reduce_sch.h` + 规约原语 `Vec::ReduceSumOp` 不在仓**（同 add 的 BroadcastSch 形态）；**tiling 计算器在仓**（op_common，1293 行）——分核/BLOCK/跨核归约事实有一手锚点。AIC 核早退（apt.cpp:31-33），AIV-only。
>
> **🔺 低价值预判：非低价值——P2.5 降级执行①编译门+②正确性门，P4 不豁免。**
> - **规约家族首卡**（reduce_mean/max/min/prod 同族），家族代表性本身即提取价值；
> - **❌ 候选 ≥2**：① `groupR>1` 跨核归约——GM workspace（vectorCoreNum × align(outSize×bytes, cacheLine)）+ ScheduleMode=1 核间同步两阶段归约，Triton 单 kernel 无核间同步，须 wrapper 两段式或单核全载降级；② `useNddma` 非连续 NDDMA 搬运无 Triton 对应（性能机制，无语义影响）；
> - **属性/语义面**：axes 任意组合（tensor 输入 i32/i64，ValueDepend）+ keep_dims + **noop_with_empty_axes**（空 axes 直返 x）+ dtype promote（bool→int64；bool 且 reduce 元素数 <2²⁴ 时经 FLOAT 中间域——aclnn:261-268）+ 非连续双路径；
> - **后端敏感点**：`tl.sum`→`linalg.reduce` T2 [SRC tl_op_lowering.md:39]；**i64 vreduce 在 A2/A3 标量化**（dtype_support.md:58/68 [SRC]，950 路径不同勿外推）——i64 输入直算 vs 降域需①门取证；
> - 联动结论：**走 P2.5 降级路径（①+②）**，BLOCK 选型与瓶颈定型两态，confidence 不升 measured(skeleton)。

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/reduce_sum_def.cpp` | ✅ | x：{F16,F32,BF16,I32,I64,BOOL}×2 组（:19-21）；**y：BOOL→I64，其余同型**（:23-25）；axes：{I32,I64} ValueDepend OPTIONAL（:36-38）；Attr **keep_dims** 默认 false、**noop_with_empty_axes** 默认 true（:40-41）；DynamicCompileStatic/Rank/Shape；**ascend950 + ascend350 双平台**（:53-54） | §2、§3 |
| kernel 入口（壳） | `op_kernel/reduce_sum_apt.cpp` | ✅ | AIC 早退（:31-33）；workspace 必需（:34-40）；`PromoteType = bool→int64_t，否则 GetPromoteType<DTYPE_X>`（:45）；bool/非 bool DAG 分派（:47-54） | §4.2 |
| **计算 DAG（核心语义资产）** | `op_kernel/arch35/reduce_sum_dag.h` | ✅ | 非 bool：`CopyIn<T>→Cast<PromoteT>→ReduceSumOp<PromoteT>→Cast<T>→CopyOut`（:28-38）；**BoolDag：i8→half→float→int64 三级 Cast 后 ReduceSumOp<int64_t>**（:40-52）；MemCfg LEVEL_2 | §6 |
| 调度器本体 | `atvoss/reduce/reduce_sch.h`、`reduce_operator.h`（Vec::ReduceSumOp 原语） | ❌ **不在仓** | 半壳缺省项——Reg 级规约展开、DB 流水细节不可见；DAG 声明 + tiling 参数反推 | §6、§9 |
| **base 库 tiling 计算器** | `model/opbase-master-src-op_common/src/op_common/atvoss/reduce/reduce_tiling.cpp`（1293 行）+ `reduce_tiling_batch_invariant.h` | ✅ | 见下「tiling 事实锚点」 | §4.1、§5 |
| tiling key | `op_kernel/arch35/reduce_sum_tiling_key.h` | ✅ | `REDUCE_TPL_KEY_DECL_FULL` 宏展开（声明在缺失头文件，登记） | §4.1 |
| GEIR 原型 | `op_graph/reduce_sum_proto.h` | ✅ | 登记用 | §3 |
| **aclnn 层** | `op_api/aclnn_reduce_sum.cpp`（325 行） | ✅ | dtype 参数=输出类型，self≠dtype 先 Cast 后 reduce（:245-249）；**空 dims→全部轴**（:252-260）；**bool+int64 特判：reduce 元素数 <2²⁴(16777216) 经 FLOAT 中间域，否则 INT64**（:261-268）；非连续支持路径（view+ReduceSum+Cast+ViewCopy，:177-189, 269-284）vs Contiguous 化路径 | §2.2、§6 |
| 文档 | `README.md` | ✅ | 「返回给定维度中输入张量每行的和」；shape 0-8 维；非连续 ND；**确定性计算：默认确定性实现**；平台表（A2/A3/950 ✓，200I/500 A2 ✗）；Atlas 推理/训练系列 dtype 含 DOUBLE/COMPLEX64/128（=仓外实现，本卡不管） | §2、§3 |
| golden | `tests/assets/golden.py` | ✅ | **f16/bf16 升 fp32 求和后回原型**（:40-47），其余 np.sum 直算；axes None → noop 直返 x（noop_with_empty_axes=True 默认）/ 全轴（False） | §测试设计 |
| ST | `tests/st/arch35/ttk_kernel_reduce_sum_st.csv` + `tests/st/aclnnReduceSum/atk_aclnnReduceSum.json` | ✅ | kernel 层 ST + aclnn 层 atk 双套 | §测试设计 |
| UT | `tests/ut/op_host/arch35/test_reduce_sum_tiling.cpp` + infershape + api + **aicpu** | ✅ | AICPU 兜底存在（ut/op_kernel_aicpu） | §测试设计 |

## tiling 事实锚点（reduce_tiling.cpp，全部在仓）

- **形状预处理**（TransformShape :613-620）：EliminateOne 去 1 维 → PadDimForNonContiguous → MergeAxis 合轴（相邻同 A/R 属性且连续维合并）。
- **Pattern 分派**（DoTilingMatchPattern :1262-1290）：按预处理后维数 1~9 → **A / AR / ARA / ARAR / … / ARARARARA**（A=规约轴、R=保留轴交错），>4 维 pad 1 到 8/9 维；A 轴判定 = 奇偶位（IsAxisA :284-291）。
- **5 步 ComputeTiling**（:1061-1091 注释）：① 按 DAG buffer 需求算 **basicBlock**（UB−CACHE_BUF_SIZE−reserved，reduce 前开 DB ×2，cast 前后比例 ratio 折算，FloorAlign 到 vRegSize；resultBlock clamp [MAX_INNER_A, CACHE_BUF_SIZE] 且 cacheLine 对齐，:951-1044）；② 按 cacheLine 找切分轴（ComputeCacheLineBlock :649+，例：f32 shape(2,35,7) cacheLine 256B → axis1 step10 outer4）；③ UB 内 A 轴切分（最大 512B 或 A 轴分核低于 85%）；④ UB 内 R 轴切分；⑤ R 轴过小回调 A 轴（上限 UB 二分缓存）。
- **分核**（SetTilingData :1101-1140）：总块数 unitA.outer×unitR.outer 均分 vectorCoreNum；**A 轴优先分核**（factorACntPerCore=ceil(outerA/numBlocks)，numBlocks 对齐到 outerA 的倍数 capped by 核数）；R 轴次之（factorRCntPerCore）；realCoreNum 重算。
- **跨核归约**：`groupR = ceil(outerR/factorRCntPerCore)`；**groupR>1 → ScheduleMode=1（核间同步）+ GM workspace = vectorCoreNum × align(outSize × maxInputBytes, cacheLine)**（SetTilingData :1123-1126 + CalcUserWorkSpace :1249-1259）——部分和落 GM 后二阶段归约（调度器细节在缺失本体，机制由 workspace 尺寸反推）。
- **空 tensor 特判**（ComputeEmptyTiling :1047-1058）：去 R 轴按全 A pattern 切，outSize=0 直返。
- **batchInvariant 分支**（:1074-1079）：R 优先切分（ComputeRFirst），尾轴 A=1 保留标识（isTailAOne，GetInputParam :192-195）——确定性场景专用。
- **all-reduce 快路径**（GetInputParam :213-219）：输出标量且输入连续 → 不读 axes 常量，全轴规约。
- **useNddma**（SetTilingData :1137）：非连续场景的 NDDMA 搬运判定。

## 语义速览（P1 详抽对象）

`y = Σ x over axes`，多轴任意组合；keep_dims 控制输出维保留；**空 axes 默认 noop 直返 x**（noop_with_empty_axes=True）；dtype 参数独立指定输出类型（aclnn 层 Cast+reduce；GEIR 层 y dtype 由 def 表定——bool→int64）。计算域：非 bool 经 PromoteType（GetPromoteType，定义在缺失头——golden 佐证 f16/bf16→fp32；i32/i64 待查，登记缺省）；bool 经 i8→half→float→int64 链（aclnn 层另有限流：reduce 元素 <2²⁴ 走 FLOAT 中间域）。

## 缺省项与注意点（P1 必须处理）

1. **半壳**：`ReduceSch` 调度器 + `Vec::ReduceSumOp` 原语 + `GetPromoteType` 定义 + `REDUCE_TPL_KEY_DECL_FULL` 展开 + 常量（CACHE_BUF_SIZE/MAX_INNER_A/vRegSize）均不在仓——DAG 声明 + tiling 参数 + golden 三方互证，不可见部分在卡片标「机制推断」。
2. **跨核归约不可直译**：groupR>1 的 GM workspace 两阶段 + ScheduleMode=1 核间同步——Triton 单 kernel 无核间同步；降级候选：① wrapper 两段式（partial kernel + final kernel）；② 每输出元素单 program 全载 R（A 轴全载语义，groupR=1 形态）。P2 给判定树。
3. **i64 规约后端敏感**：vreduce i64 A2/A3 标量化 [SRC]，950 路径不同——本栈 910_9372 实测前勿外推性能；正确性不受限。①门附带取证。
4. **bool 输入**：y=int64，DAG 三级 cast 链 + aclnn 2²⁴ 限流——Triton 侧 wrapper 直接 zeros/astype 语义对齐即可，但须登记精度域差异。
5. **noop_with_empty_axes**：空 axes 直返 x（identity）——wrapper 早退，不进 kernel。
6. **非连续输入**：aclnn 双路径（view 直算 vs Contiguous 化）；Triton wrapper 按 contiguous 接（torch.sum 语义层已处理），登记。
7. axes 是 tensor 输入（ValueDepend）→ wrapper 按 Python int/tuple 标量接，**kernel 内 axes 相关 shape/stride 运行传参**（C-017：stride 不 constexpr 化）。
8. README 平台表 dtype 含 DOUBLE/COMPLEX（Atlas 推理/训练系列）= 仓外实现，本卡只管 def 表 6 dtype × 950/350。

## P0 质量门结论

材料齐（半壳缺省项已识别并有互证方案）→ **通过，进入 P1**。判**非低价值**（规约家族首卡、❌≥2、属性/特殊语义面、i64 规约后端未取证）→ **P2.5 降级执行①编译门+②正确性门**，P4 不豁免。
