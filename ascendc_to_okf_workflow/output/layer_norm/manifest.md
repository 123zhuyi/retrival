# P0 材料清单 — layer_norm（layer_norm_v4）

> 生成时间：2026-09-18｜来源目录：`src/ops-nn/norm/layer_norm_v4/`
> **kernel 类型判定：AIV-only 归一化算子（半壳）**——arch35 regbase 三家族 kernel 中：TWO_PASS 用 **`AscendC::LayerNorm` 高阶 API**（本体不在仓）、WELFORD 用 **`AscendC::WelfordUpdate/WelfordFinalize/Normalize` 向量内置函数/库 API**（实现不在仓，tiling 经 `GetWelfordUpdateMaxMinTmpSize` 等取 tmp 尺寸）；TWO_PASS_PERF 为裸向量指令全开实现（在仓 643 行）；**host 侧三模板 tiling + base/common tiling 全部在仓**；另有一份 630 行设计文档（Welford 递推推导 + 二分累加 + 模板选择策略，金矿）。
>
> **🔺 低价值预判：非低价值——P2.5 降级执行①编译门+②正确性门，P4 不豁免。**
> - **Norm 家族首卡**（rms_norm/group_norm/instance_norm 同族同构），家族代表性本身即提取价值；
> - **❌ 候选 ≥3**：① Welford 向量化内置（`WelfordUpdate` 64 f32/SIMD 并行递推）无 Triton 直接对应——须以块级 merge（Chan 并行合并公式）或 two-pass 重写；② 二分累加树（binaryAddQuotient/binaryAddK，O(log n) 误差）是硬件向量指令级优化——Triton 侧 `tl.sum` 本身是树形归约（天然同构语义收益），但装载/归并结构不可逐式译；③ FusedMulDstAdd 融合指令/行配对/ShiftLefts 对齐等 perf 优化无对应；④ membase(910b) 六模板 + arch35 三模板的多变体 tilingkey 分派体系整体不可译，只提取选择策略的语义面；
> - **属性/语义面**：normalized_shape（尾部任意多轴折叠 R）+ gamma/beta OPTIONAL（nullptr 语义=恒 1/恒 0）+ epsilon attr（默认 1e-5）+ **mean/rstd 恒 f32 双输出** + 空 tensor 特判（M=0 早退 / N=0 时 mean=0、rstd=NaN）+ implMode（aclnn 层 HIGH_PRECISION/HIGH_PERFORMANCE/KEEP_FP16，仅 V3 路径消费）；
> - **后端敏感点**：`tl.sum`/归约 T2 [SRC]；f16/bf16 升 fp32 域（与 sum 同纪律）；rsqrt lowering（`tl.rsqrt` → 待查 tl_op_lowering.md，①门取证）；输出 mean/rstd 是**每行一个标量**的跨 program 写（行主语形态，无 atomic 需求）。
> - 联动结论：**走 P2.5 降级路径（①+②）**，BLOCK 选型与瓶颈定型两态，confidence 不升 measured(skeleton)。

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/layer_norm_v4_def.cpp` | ✅ | x 五组 {f32,f16,f16,bf16,bf16} 对应 gamma/beta dtype 组合；normalized_shape 为**输入**（i32 tensor）非 attr；gamma/beta OPTIONAL；y 同 x；**mean/rstd 恒 f32**；Attr epsilon 默认 1e-5；平台 910b/910_93/310p + config json（ascend950/910_93/910b/310p/kirin） | §2、§3 |
| infershape | `op_host/layer_norm_v4_infershape.cpp` | ✅ | y=x shape；mean/rstd shape = x 去尾部 normDim 维（arch3510 下尾部维补 1，aclnn 校验） | §2 |
| kernel 入口（apt 壳） | `op_kernel/layer_norm_v4_apt.cpp` + `_apt.h` | ✅ | 23 行壳→`layer_norm_impl`；**arch35 regbase 三家族分派：3xx=TWO_PASS(300/310/311/320/322)、4xx=WELFORD(400/410/411/420/422)、5xx=TWO_PASS_PERF(500/510/511/520/522)**；模板参 `<Tfm, Tweight, Tmean>`（f32/ff、f16+fp32w、f16 全半、bf16+fp32w、bf16 全半） | §4.1 |
| kernel 入口（membase 壳） | `op_kernel/layer_norm_v4.cpp` | ✅ | 910b 家族：100s=SINGLE_READ、210/211+LNV4_TRANSPOSE_*=TRANSPOSE、700s=GENERAL、600s=ALIGN、1600s=ALIGN_LIMIT、2600s=ONE(merge_n)——**910_9372 目标平台走 arch35，本家族裁剪登记不抽** | §4.1（裁剪） |
| TWO_PASS kernel | `op_kernel/arch35/layer_norm_v4_two_pass.h` | ✅ | x 折 (A,R)，每核 aBlockFactor 行，UB 循环 aFactor 行；计算核心 = **`AscendC::LayerNorm` 高阶 API（本体不在仓，半壳成分）**；外层含二分累加 `CalculateMeanVarVF`（binaryAddQuotient/binaryAddK，见设计文档 §4）；epsilon 成员默认 1e-5(:337)；mean/rstd GM 偏移=aBlockFactor×blockIdx(:98-99) | §5、§6 |
| WELFORD kernel | `op_kernel/arch35/layer_norm_v4_welford.h` | ✅ | Process=WelfordInitialize→ProcessWelfordUpdate(tile 循环+tail)→WelfordFinalize(M2/N→var→rstd)→ProcessNormalize 二遍（**统计 1 遍+归一化 1 遍=2 次数据访问**）；`WelfordUpdate` 向量内置 64 f32/lane（实现不在仓）；tileLength=UB 约束最大 64 对齐值 | §5、§6 |
| TWO_PASS_PERF kernel | `op_kernel/arch35/layer_norm_v4_two_pass_perf.h` | ✅ | **全开实现（在仓 643 行）**：第一遍算 mean 同时存 (x−mean) 到 UB（省一次 GM 搬移，3 次→2 次）；第二遍直接对差值平方累加得 var；CalculateNormalizeVF **行配对两行 + `FusedMulDstAdd` 融合指令**（dst=src*gamma+beta 单周期）；`LAST_LOOP_NUMS` 模板参控制末级归并（1=ReduceSum / 2=ShiftLefts 指数对齐+Add） | §5、§6 |
| regbase_common | `op_kernel/arch35/layer_norm_v4_regbase_common.h` | ✅ | 公共 tiling 结构访问（85 行） | §4 |
| tiling base | `op_host/layer_norm_v4_tiling_base.cpp` | ✅ | commonParams：colSize=**外层行数 M**、rowSize=**归一化轴长 R**（tiling_base:160-168 折叠）、rowAlign=32B 对齐、eps、isRegBase、vlFp32=vectorLength/4；**dtypeKey=tensorKey×10+paramKey**（f32=0/f16=1/bf16=2，:33-54）；normalized_shape 必须 0D/1D 且 ≤ x 维数（:139-152） | §4.2 |
| WELFORD tiling | `op_host/layer_norm_v4_welford_tiling.cpp` | ✅ | **注册优先级 4000（最高）**；IsCapable=isRegBase；M 按 coreNum 均分 blockFactor；DoLibApiTiling 从 UB 预算反推最大 tileLength（步进 64、双缓冲、mean/rstd/临时 buffer 全计）；workspace 16MB 常量 | §4.2 |
| TWO_PASS tiling | `op_host/layer_norm_v4_regbase_two_pass_tiling.cpp` | ✅ | 优先级 200；IsCapable=isRegBase+UB 二分搜 aFactor（CanFitInBuffer）；**binaryAddQuotient：从 vlFp32=64 翻倍至 ≥R 再除 2**（:63-67）；powerOfTwoForR=≥R 最小 2 幂（:131-134）；**binaryAddK 归并层数**（vcaddNum>64 时，:148-162）；tilingKey=300/310/311/320/322 按 dtype 组合 | §4.2 |
| TWO_PASS_PERF tiling | `op_host/layer_norm_v4_regbase_two_pass_perf_tiling.cpp` | ✅ | 优先级 150；IsCapable 最严：**rowAlign>2·vlFp32²·2=16384 拒绝（:63）**；**colSize≥4096 且 rowAlign≥8192 联立拒绝（:68，LN_ROW_THRESHOLD/LN_COL_THRESHOLD）**；UB 校验含 (x−mean) 差值 buffer | §4.2 |
| common tiling（非 regbase 兜底） | `op_host/layer_norm_v4_common_tiling.cpp` | ✅ | 优先级 2000 但 IsCapable=`!isRegBase && !is310P && !isV3`（与 regbase 互斥）；tilingKey=700+dtypeKey%10；membase GENERAL 对应 | §4.2（裁剪） |
| 其余 host tiling | single_read/merge_n/transpose/tiling.cpp | ✅ | 910b 家族用，裁剪登记 | §4.1（裁剪） |
| **aclnn 层** | `op_host/op_api/aclnn_layer_norm.cpp`（512 行） | ✅ | dtype 支持表（910:{f16,f32}、910B+:{+bf16}，:53-57）；weight/bias 非 f32 时必须与 input 同型（:131-141）；**regbase 恒走 LayerNormV4**，非 regbase 按 IsV4SocCheck（bf16 同型/310p f16/910b 非对齐小轴）选 V4 否则 V3（:276-293）；**空 tensor：M=0 直返；N=0 时 mean=0、rstd=NaN**（:354-374）；weight/bias 缺失时非 regbase 填 1/0，regbase 传 nullptr（:380-414）；输出 Cast+ViewCopy | §2.2、§6 |
| **设计文档（金矿）** | `docs/ParallelWelford_and_BinaryAccumulation_LayerNormV4.md`（630 行） | ✅ | §3 Welford 递推推导（delta=x−mean; mean+=delta/n; M2+=delta·(x−mean_new); var=M2/n，避免 E[X²]−E[X]² 灾难性抵消；f16/bf16 下更显著）；§4 二分累加（quotient 拆分+二叉归并树，误差 O(n)→O(log n)；为何 Welford 不能替代：M2 累加器 UB 占用）；§5 模板优先级 4000/200/150 与 IsCapable 三维权衡（Welford 牺牲 UB / TwoPass 牺牲带宽 / Perf 牺牲适用面） | §3、§5、§6 |
| golden / ST / UT | `tests/` | ✅ | 登记，P2.5 ②门对 torch 语义基准 | §测试设计 |

## 语义速览（P1 详抽对象）

`y = (x − mean) × rstd × gamma + beta`，`rstd = 1/sqrt(var + eps)`，沿尾部 normalized_shape 折叠轴 R 归约；mean/rstd 恒 f32 按行输出（shape=x 去尾部 normDim 维）。gamma/beta OPTIONAL（nullptr=恒 1/恒 0，regbase 直传 nullptr 标志）。dtype 五组：f32/ff、f16+f32w、f16/ff16、bf16+f32w、bf16/fbf16（paramDtype 独立）。算法族三选：**Welford（单遍统计，精度最优，M2 占 UB）→ TwoPass（两遍统计+二分累加，超长轴兜底）→ TwoPassPerf（两遍但 (x−mean) 复用+行配对融合，中等轴极致性能）**，host 按 IsCapable 逐级筛选。计算域：统计与归一化在 fp32 域（f16/bf16 全半型时 gamma/beta 乘加在半型域还是 fp32 域——P1 详查 two_pass_perf.h 的 cast 点）。

## 缺省项与注意点（P1 必须处理）

1. **半壳成分**：`AscendC::LayerNorm` 高阶 API（two_pass 计算核心）、`WelfordUpdate/WelfordFinalize/Normalize`（welford 计算核心）、`Get*MaxMinTmpSize` 系列——实现不在仓；two_pass_perf 全开可作语义参照，golden/torch 作真值。卡片对库 API 内部标「机制推断」。
2. **目标平台裁剪**：910_9372=arch35 regbase——membase(910b) 六模板（single_read/transpose/general/align/align_limit/one）+ common_tiling 兜底家族整体裁剪登记，不抽取。
3. **Welford 的 Triton 同构选择**：Triton 无逐元素递推向量内置；候选 ① 块级 Chan 并行合并（每 tile 算 (n,mean,M2) 再合并，数学同构）；② in-kernel two-pass（sum→mean→sum((x−mean)²)→rstd，tl.sum 树形归约天然具备二分累加的 O(log n) 误差特性）——P2 给判定树，倾向 ②（结构简单且精度等价论证充分）。
4. **R 轴超长**：官方 TwoPass 存在即因 R 大时 UB 放不下 M2 数组；Triton 单 program 行循环 tile 化天然不受此限（寄存器/本地内存压力另论），BLOCK_N 选型两态。
5. **rsqrt lowering**：`tl.rsqrt`（或 1/tl.sqrt）950 后端 lowering 未取证——①门附带验证。
6. **mean/rstd 双输出**：每行一个 f32 标量，行主语 program 直写，无 atomic；wrapper 需支持「不需要 mean/rstd」的早退（torch layer_norm 常不取）——可选输出按 flag constexpr 化。
7. **空 tensor 语义**：M=0 早退；N=0（归一化轴空）→ y 空、mean=0、rstd=NaN——wrapper 处理，不进 kernel。
8. **非连续输入**：aclnn Contiguous 化；Triton wrapper 按 contiguous 接（torch 语义层处理），登记。
9. epsilon 是 attr（f32），运行传参；**gamma/beta 存在性 constexpr 化**（HAS_GAMMA/HAS_BETA，C-023 同纪律）。
10. dtypeKey 组合中 f16+f32w/bf16+f32w（混合精度权重）是独立模板——Triton 侧 wrapper 按 gamma dtype 分支传 constexpr 码，kernel 内 cast 点 P1/P2 定。

## P0 质量门结论

材料齐（半壳缺省项已识别：两个库 API 家族有 two_pass_perf 全开实现 + 设计文档 + torch 真值三方互证）→ **通过，进入 P1**。判**非低价值**（Norm 家族首卡、❌≥3、属性/特殊语义面丰富、rsqrt 后端未取证）→ **P2.5 降级执行①编译门+②正确性门**，P4 不豁免。
