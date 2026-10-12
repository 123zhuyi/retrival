# P1 语义抽取 — cumsum

> 生成时间：2026-09-17｜依据：`output/cumsum/manifest.md`（**非低价值** → P2.5 降级①+②，P4 不豁免）
> 根目录 `src/ops-math/math/cumsum/`；sklansky 文件均指 `op_kernel/arch35/cumsum_base/`。

## 1. 功能与公式

- `y_i = x_1 + … + x_i` 沿指定 dim 轴前缀和（README）；输出 shape = 输入 shape，dtype 与 x 相同。
- 属性：**exclusive**（排除自身：y_i = Σ_{j<i} x_j，首元 0）、**reverse**（从尾部累加），默认均 false（def.cpp:67-68）。
- golden 语义（golden.py）：np.cumsum 主线；**exclusive = roll(1)+首位置 0**；**reverse = flip→cumsum→flip**；f16/bf16 升 fp32 计算后回原型。

## 2. 接口与原型

### 2.1 GEIR op（cumsum_def.cpp）
- x：8 型 {INT8, INT32, INT64, UINT8, UINT64, F16, F32, BF16}，ND（:24-36）。
- **axis：tensor 输入**（int32/int64 各 8 配型，REQUIRED + ValueDepend OPTIONAL，:38-52）——不是属性；host tiling 值依赖读轴号。
- y 与 x 同型（:53-65）；exclusive/reverse OPTIONAL bool（:67-68）；DynamicRank/Shape（:73-74）；`opFile=cumsum_apt`（:76）；**仅 ascend950**（:77）。

### 2.2 aclnn 层（两套 API 语义）
- **aclnnCumsum（V1）= torch.cumsum 语义**：dim 为标量属性（ST：torch.cumsum，202 case）。
- **aclnnCumsumV2 = tf.cumsum 语义**：axis tensor + exclusive/reverse 属性（ST：tf.cumsum，200 case）——两属性仅 V2 暴露，但 kernel 两属性全支持（def 层）。
- AICPU 兜底注册（op_kernel_aicpu/），非 NPU AICore 主路径。

## 3. 规格约束

| 项 | 值 | 锚点 |
|---|---|---|
| dtype | i8/i32/i64/u8/u64/f16/f32/bf16，in/out 同型 | def.cpp:26-35,55-64 |
| axis | tensor（i32/i64 标量），值依赖 | def.cpp:38-52 |
| 属性 | exclusive/reverse bool，默认 false | def.cpp:67-68 |
| 平台 | ascend950（AICore）；AICPU 兜底另册 | def.cpp:77 |
| ST 值域 | V1：nd(mean[-100,100],std[1,25])；V2：[-7,7] | 两套 atk json |

## 4. 分核策略与 tiling（✅ 全部在仓，host 决策树）

### 4.1 问题折叠与决策树（cumsum_tiling_ascendc_arch35.cpp）
- 折叠 **M×R×N**：M=dim 外维乘积、R=扫描轴长、N=dim 内维乘积（contiguous）。
- 决策轴：M/R/N vs coreNum、cache line 512B、UB 容量、VL：
  1. **N ≥ CL 或 R 可全载**：M 分核（M 不够→借轴 N）；R 全载 UB → **1001/1002**；R 超 UB → **1011/1012（UB_SS）**（:449-516, 564-574）。
  2. **oneway vs twoway（JudgeSklanskyPatten :1084-1099）**：alignN > VL/4 → oneway；否则 DoFold（foldCount≤8，fp16 限 FP16_FOLD_MAX）后 foldCount>1 → twoway。
  3. **M 借 N 后仍不够核 → 借轴 R → CORE_SS（1101/1102）**；R 再超 UB → **1111/1112**（:534-560, 774-840）。
- UB 预算项：UB_SS_BUFFER_SIZE/BR_BUFF/BR_IDX_BUFF 等常量从 ubSize 扣除（:487-489）。

### 4.2 整型家族决策（cumsum_tiling_ascendc_int_arch35.cpp:315-322）
- R 跨核（isRBlockAxis）→ **CUM_WITH_GROUP 11002**（MIX_AIV_1_0）；
- N·dtypeSize ≥ VL → **CUM_NO_SPLIT 11000**（单核整条线）；
- N 窄 → **CUM_AR_SPLIT 11001**（核内拆 R 提并行）。
- 整型 kernel 要求 workspace 非空（cumsum_apt.cpp SetSysWorkspace）。

### 4.3 tilingdata
- `CumsumSklanskyTilingData`（浮点：lenM/lenR/lenN、各轴 blockPara/ubPara、nBorrow/rBorrow、sklanskyItersTiling[6]、exclusive/reverse、xBufSize 等）；`Cum4IntTilingData`（整型）。

## 5. 片上内存

- 浮点：xBuf（ubBufferSize）+ cast 后 fp32 buffer（非 fp32 输入时同尺寸 fp32，oneway_sklansky.h:88-93）；CORE_SS 另加 addFactor/addData/addFactorCasted/addDataCasted/addFactorOffset 5 个 TBuf（core_sklansky.h:71-77）。
- 跨核暂存直接复用 **y（GM）自身**做读-改-写，无额外 workspace（浮点家族）。

## 6. 计算语义（核心资产）

### 6.1 Sklansky 单向扫描（oneway_sklansky.h）
- CopyIn：R 轴行 × N 列搬入 UB，行内 dupSize=32B 对齐、可 mFold 倍复制凑向量（:115-135）。
- **Compute：log2(rFactor) 趟**（:274-281）。趟 k（addLoopPow=2^k）：每 2^(k+1) 行一组，pivot=组内前半末行（(2^k−1) 行），将 pivot **扇出加**到组内后半 2^k 行（ComputeInner :256-272 + ComputeWithFlag :196-254，MicroAPI LoadAlign/Add/StoreAlign 寄存器级）。
- **reverse**：`flag=−1`，pivot/目标地址反向走（:271）——同一算法镜像。
- **exclusive**：CopyInExclusive 整体错位一行 + 边界 Duplicate(0)（:138-177）——即「右移一位、首位置 0」，与 golden roll 语义一致。

### 6.2 twoway 折叠（twoway_sklansky.h）
- N 窄（≤VL/4，MAX_N=32）时把多行 R 折叠进同一向量寄存器组（MAX_FOLD=8），先对折叠后大行做 Sklansky 再展开修正——**纯布局/向量宽度工程，数学语义与单向一致**。

### 6.3 UB_SS（核内分块进位，cumsum_ub_ss_*.h）
- R 超 UB → R 分块，块内 Sklansky，块间 carry（前块总和）加到后块全部行——顺序依赖在核内消化。

### 6.4 CORE_SS（跨核两级，core_sklansky.h —— ❌ 核心）
- 各核先对本地 R 段完整扫描+写回 y（外层 cumsum_core_ss_oneway_ss.h:41-68 复用 oneway handler）。
- 然后 **log2 轮跨核 Sklansky 合并（SS_ITER_MAX=6）**：每轮 `SyncAll()` 全局屏障（core_sklansky.h:120-122）→ 从 GM(y) 读 addFactor（前一组末行）→ 加到本组各行 → 写回 GM。
- 语义 = 分段扫描 + 段间前缀和传播的并行扫描标准两阶段，但实现依赖**核间全局屏障 + GM 读-改-写**。

### 6.5 整型家族（cumsum_4_int.h）
- 三个独立类（CumNoSplit/CumSplitAR/CumWithGroup），原生整型精度（GetPromoteType 整型=自身，cumsum_base.h:79-103）；含 bank group conflict 规避（:1220 注释）。语义与浮点家族同构（单核整线/核内拆/跨核组）。

### 6.6 精度策略
- f16/bf16 → **fp32 中间域**（Cast NONE 进 / RINT 出，oneway CopyInDataCast :180-193）；fp32 直通；整型原生。
- Sklansky 树形求和顺序 ≠ np.cumsum 顺序累加——fp 下与 golden 存在舍入路径差（ST acc=default 容差覆盖）。

## 7. 数值精度

- fp32 中间域必须保留（f16/bf16）；RINT=RTNE 回写。
- 树形归约顺序差：长 R 时与顺序 golden 的误差随 log2(R) 增长而非 R——通常**优于**顺序累加，但比对基准仍是 golden，容差按 ST default。
- exclusive 的代数实现（inclusive − x）在 fp 下 ≤1ulp 差（golden 是精确移位）——P2 决策点。

## 8. 测试与标杆

- golden：§1（np.cumsum + roll/flip 变体，f16/bf16 升 fp32）。
- ST：V1 202 case（torch.cumsum，dim 标量）+ V2 200 case（tf.cumsum，exclusive/reverse 组合）；acc=default。
- UT：host tiling + AICPU。
- 无仓内 benchmark 脚本。

## 9. 偏差与风险清单

1. **跨核 SyncAll 协议不可翻译**（§6.4）——Triton 单 kernel 无全局屏障；降级方案 = 单 program 整线 / 两 kernel 分段，P2 详述。
2. **tl.cumsum lowering = func::CallOp**（tl_op_lowering.md:70 [SRC]）——性能不透明；备选纯加法行序循环。P2.5 ①编译门实测。
3. vcum* dtype：i64 标量化；A2/A3 非 i64 且 cum 维最内也标量化（dtype_support.md:165 [SRC]）——950 1D dim0 i32→i64 走 SIMT 库例外。
4. 树形求和 vs golden 顺序求和的舍入路径差（§7）——容差内，登记。
5. axis tensor 值依赖 → host tiling 需 axis 可知；动态 axis（运行时变）场景 host 有兜底？未确认，登记（AICPU 可能承接）。
6. AICPU 兜底触发面未定位——本卡只管 AICore 路径。

## P1 质量门

- [x] 三层语义齐备（kernel 四策略家族/aclnn 两版/测试层）
- [x] 关键论断带 file:line 锚点；全部实现仓内一手
- [x] ❌≥3 逐项识别（跨核屏障/Reg Sklansky/布局折叠），均有降级方向
- [x] **非低价值确认 → P2.5 降级①+②**，P4 不豁免
