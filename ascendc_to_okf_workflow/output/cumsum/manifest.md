# P0 材料清单 — cumsum

> 生成时间：2026-09-17｜来源目录：`src/ops-math/math/cumsum/`
> **kernel 类型判定：AIV-only scan（前缀和）算子，双家族**——浮点家族（f16/f32/bf16，PromoteType=fp32）8 个 tilingkey 的 Sklansky 扫描变体 + 整型家族（i8/u8/i32/i64/u64）3 个 tilingkey 独立实现（`cumsum_4_int.h` 1651 行）；跨核档 `KERNEL_TYPE_MIX_AIV_1_0`。另有 AICPU 兜底（`op_kernel_aicpu/`）。**非壳算子**：全部调度/计算本体在仓（~6800 行），是本批三个算子中语义密度最高的。
>
> **🔺 低价值预判：非低价值——P2.5 执行（降级①编译门+②正确性门），P4 不豁免。**
> - **❌ 项 ≥3**：① 跨核 `SyncAll()` 协议（core_ss 档，GM 暂存 + log2(rBorrow) 轮全局屏障）Triton 无对应；② Reg 级 Sklansky 扇出（LoadAlign/StoreAlign 寄存器直写，log2(R) 趟）无对应；③ mFold/N-fold UB 内布局折叠（twoway 档，MAX_FOLD=8）无对应。
> - **后端陷阱已取证**：`tl.scan`（cumsum）lowering = `func::CallOp`（库调用，非向量化直译，tl_op_lowering.md:70 [SRC]）；vcum* i64 标量化、A2/A3 非 i64 且 cum 维最内也标量化（dtype_support.md:165 [SRC]）。
> - 有 scan 语义（非纯 elementwise）；有属性面（exclusive/reverse）；axis 为 tensor 输入（ValueDepend）。
> - 联动结论：**走 P2.5 降级路径（①+②）**——BLOCK 选型与瓶颈定型论断保持两态，confidence 不升 measured(skeleton)；P4 正常执行（不降为冒烟）。

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/cumsum_def.cpp` | ✅ | x：8 型 {INT8,INT32,INT64,UINT8,UINT64,F16,F32,BF16}（:24-36）；**axis = tensor 输入**（int32/int64，REQUIRED+ValueDepend OPTIONAL，:38-52）；y 与 x 同型（:53-65）；**Attr exclusive/reverse OPTIONAL bool 默认 false**（:67-68）；DynamicRank/Shape（:73-74）；`opFile=cumsum_apt`（:76）；仅 ascend950（:77） | §2、§3 |
| GEIR 原型 | `op_graph/cumsum_proto.h`、`cumsum_graph_infer.cpp` | ✅ | 登记用 | §3 |
| kernel 入口 | `op_kernel/cumsum_apt.cpp` | ✅ | 浮点 8 key（:13-20）：1001/1002=oneway/twoway（R 全载 UB）、1011/1012=UB_SS（R 分块+核内进位）、1101/1102=CORE_SS（R 跨核）、1111/1112=CORE_SS+UB_SS 复合；整型 3 key（:22-24）：11000 NO_SPLIT / 11001 AR_SPLIT / 11002 WITH_GROUP（MIX_AIV）；`cumsumSimd` AIV-only（:29）、`cumsumSimdInt` 要求 workspace 非空（SetSysWorkspace） | §4.2、§6 |
| **Sklansky 扫描核心** | `op_kernel/arch35/cumsum_base/cumsum_oneway_sklansky.h`（381 行） | ✅ | 问题分解 M×R×N；CopyIn 带 dupSize 32B 对齐 + mFold（:115-135）；**Compute：log2(rFactor) 趟，趟 k：组长 2^(k+1)，pivot=组内前半末行，扇出加到后半 2^k 行**（ComputeInner :256-272 + ComputeWithFlag :196-254）；reverse 用 flag=−1 反向走（:271）；exclusive 走 CopyInExclusive 错位+Duplicate(0)（:138-177） | §6.1 |
| twoway（折叠变体） | `cumsum_base/cumsum_twoway_sklansky.h`（587 行） | ✅ | N 窄（≤vRegSize/4）时把多行 R 折进单个向量寄存器组：MAX_N=32、MAX_FOLD=8、VL_ELEM（:40-47）；foldCount 由 DoFold 按 vRegSize 迭代（tiling:1101-1116） | §6.2 |
| UB_SS（核内分块进位） | `cumsum_ub_ss_oneway_ss.h`（149 行）等 | ✅ | R 超 UB 时分块，块间进位（carry）传播 | §6.3 |
| **CORE_SS（跨核两级）** | `cumsum_base/cumsum_core_sklansky.h`（583 行）+ 4 个外层组合 | ✅ | **跨核 Sklansky：SS_ITER_MAX=6 轮，每轮 `SyncAll()` 全屏障（:120-122）**；addFactor=前一 R 段组末行（从 GM 即 y 读回）、addData=本轮待加行，GM 读-改-写（:60-76 结构定义）；CORE_SS 外层先各核本地扫（coreInnerHandler）再跨核合并（cumsum_core_ss_oneway_ss.h:41-68） | §6.4（❌ 核心） |
| 整型家族 | `cumsum_4_int.h`（1651 行）+ int tiling | ✅ | CumNoSplit(:1250)/CumSplitAR(:1291)/CumWithGroup(:1332)；key 选择：R 跨核→WITH_GROUP；**N·dtypeSize ≥ VL → NO_SPLIT；N 窄 → AR_SPLIT**（int tiling:315-322）；bank group conflict 规避注释（:1220） | §6.5 |
| tiling host 侧（浮点） | `op_host/arch35/cumsum_tiling_ascendc_arch35.cpp`（1390 行） | ✅ | 决策树：M/R/N vs coreNum/CL(512B)/UB；**JudgeSklanskyPatten：alignN>VL/4→oneway，否则 DoFold 后 foldCount>1→twoway**（:1084-1099）；M 够切核→M 分核；M 不够→借轴 N→再不够→**借轴 R（CORE_SS 跨核）**（:449-560 典型路径）；R 全载 UB→1001/1002，否则 UB_SS（:483-516, 655-746） | §4.1 |
| tiling host 侧（整型） | `cumsum_tiling_ascendc_int_arch35.cpp`（438 行） | ✅ | 三段轴 left/mid(R)/right；isRBlockAxis→WITH_GROUP（:315-322） | §4.1 |
| AICPU 兜底 | `op_kernel_aicpu/cumsum_aicpu.{h,cpp}` | ✅ | 不参与 NPU 主路径，登记 | §3 |
| aclnn 接口 | `op_api/aclnn_cumsum.{h,cpp}`（V1）、V2 文档 | ✅ | V1=torch.cumsum 语义（dim 标量属性）；V2=tf.cumsum 语义（axis tensor + exclusive/reverse 属性）——ST 两套 json 对应 | §2.2 |
| golden | `tests/assets/golden.py` | ✅ | np.cumsum；**f16/bf16 升 fp32 计算回原型**；exclusive=`roll(1)+首位置0`；reverse=`flip→cumsum→flip` | §测试设计 |
| ST 用例 | `tests/st/aclnnCumsum/atk_aclnnCumsum.json`（202 case）+ `aclnnCumsumV2`（200 case） | ✅ | V1：torch.cumsum，值域 nd(mean[-100,100],std[1,25])；V2：tf.cumsum 含 exclusive/reverse 属性组合，值域 [-7,7] | §测试设计 |
| UT | `tests/ut/op_host/arch35/test_cumsum_tiling.cpp`、`op_kernel_aicpu/` | ✅ | host 层 + AICPU | §测试设计 |

## 语义速览（P1 详抽对象）

**y_i = Σ_{j≤i} x_j 沿 dim 轴**（README）。问题折叠为 M×R×N（M=扫描轴外维乘积、R=扫描轴、N=内维乘积）。浮点家族全在 **fp32 中间域**（GetPromoteType：half/bf16→float，cumsum_base.h:64-104）；整型家族原生精度。

四种执行策略（tilingkey 载体，host 决策树按 shape 选）：
1. **ONEWAY/TWOWAY（1001/1002）**：R 全载 UB，单核内 log2(R) 趟 Sklansky；twoway=N 窄时折叠多行进向量组（布局工程）。
2. **UB_SS（1011/1012）**：R 超 UB → 核内 R 分块 + 块间 carry 传播（顺序）。
3. **CORE_SS（1101/1102）**：M 太小（借 N 后仍占不满核）→ **R 轴跨核切分**，各核本地扫后经 GM（y 自身）做多轮读-改-写合并，轮间 `SyncAll()`。
4. **CORE_SS_UB_SS（1111/1112）**：3+2 复合。

## 缺省项与注意点（P1 必须处理）

1. **axis 是 tensor 输入**（ValueDepend）——host tiling 需读 axis 值（值依赖）；Triton wrapper 按标量 int 接（aclnn V1 语义即标量 dim）。
2. **exclusive/reverse 两属性**：golden 语义 = roll+置首 0 / flip 两端——P2 给代数化映射（exclusive = inclusive − x；reverse = 反向寻址）。
3. **跨核 SyncAll 是 Triton 结构性 ❌**：单 kernel 内无全局屏障——映射为「单 program 负责整条扫描线」（M×N 充分并行时）或「两 kernel 分段和+偏移加」，P2 给降级论证。
4. **tl.cumsum lowering = func::CallOp**（库调用）——性能不透明，P2.5 ①编译门需实测可编译性；备选 = 纯加法行序循环（不依赖 scan op）。
5. **AICPU 兜底存在**：某些 shape/dtype 组合可能根本不走 AICore——P1 确认触发面（本仓 def 仅 950，AICPU def 另行注册）。
6. 浮点/整型家族 kernel 完全分离——Triton 单 kernel + dtype 推导覆盖，但 i64 标量化预警（dtype_support 4a/4a'）。
7. twoway/mFold 等布局折叠是寄存器宽度工程，无语义影响——不翻译，记 §6。

## P0 质量门结论

材料齐全且全部为仓内一手实现（无双底座缺省）→ **通过，进入 P1**。判**非低价值**（❌≥3、scan 语义、属性面、后端 scan 陷阱）→ **P2.5 降级执行①编译门+②正确性门**，P4 不豁免。
