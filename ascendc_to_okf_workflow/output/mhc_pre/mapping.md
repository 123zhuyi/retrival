# P2 语义转化 — mhc_pre（Ascend C → Triton，新流程版）

> 输入：extraction.md（2026-09-14 复验版）。每条 P1 事实在此有去向（转化/合并/显式丢弃+理由）。
> 可达性三态：✅ 直接可达 / ⚠️ 编译器自动或需权衡 / ❌ 不可达（含降级方案）。
> **与 2026-09-03 旧版（baseline_20260903/mapping.md）的差异**：本版按『参考源与检索指引』
> 对六类后端论断强制取证（backend_facts → zh → IR 源码），每条标 [EXP]/[DOC]/[SRC]；
> §0.5 为六任务检索记录；旧版两处错误论断 + 一处潜伏 bug 在本版修正（见 §0.6）。
>
> **核心结论（与旧版一致，予以维持）**：本算子语义 = RMSNorm(Σx²) + 瘦矩阵乘
> [T,nD]×[nD,matN]（matN≤1088）+ 分段门控后处理 + 小矩阵加权求和 hIn。Ascend C 用三种
> tiling 模式 + 跨核 flag + workspace + 两阶段解决的「K 轴分核归并」问题，在 Triton 中由
> **单 program 顺序 K 循环**天然消解——无需 split-K、原子加、跨核同步，累加顺序天然固定
> （等价 deterministic 语义）。三套模式合并为**一个 kernel**。

## 0.5 检索记录（新流程六任务，证据全标）

| # | 检索任务 | 命中 | 结论 |
|---|---|---|---|
| ① | 骨架各 tl op 的 lowering 分级 | `tl_op_lowering.md`；`dtype_support.md` 附录 A | tl.load/store = T1；**tl.dot = T2（Cube）**；tl.sum = T2（linalg.reduce）；逐元素算术/exp/div = T1（`extern_elementwise` T2）。fp32 下全部向量原生：VAdd/VMul/VDiv/VExp 仅 i64 标量化 [SRC ShouldLowerToScalarLoops.cpp]；**全 T1/T2，无慢路径** |
| ② | ⚠️/❌ 判定依据 | `backend_contracts.md` C-006/C-012；`auto_passes.md` | split-K/atomic/跨核 flag 的丢弃是**语义等价替代**（单 program 顺序 K 循环），非后端限制——论证是结构性的（无跨核写同一张量），不违反 C-012；双缓冲/预取 ⚠️ 由 multibuffer 承接（bishengir 默认 true [SRC Options.td]），**不写 num_stages**（C-006 [EXP]）；`input_precision="hf32"` ⚠️ [UNVERIFIED]，默认 "ieee" |
| ③ | 编译选项 | `compiler_options.yaml` by_operator_class.**cv_fusion** | 本 kernel 含 tl.dot → 归 CV 融合类：enable_auto_bind_sub_block / sync_solver 默认 true [SRC Options.td]，tile_mix_vector_loop/tile_mix_cube_loop 默认 2——均不需手动设；workspace=0 故 set_workspace_multibuffer 不涉及；grid 已取物理核数，auto_blockify_size 不需要 |
| ④ | UB/对齐预算常数 | `hardware_platforms.md` | **950 UB=248KB 可用**（物理 256KB 保留 8KB）[SRC NPUTargetSpec.td:223-235]——旧版按 192KB（A2 值）校核，偏保守方向安全但数值错误；UB 对齐 32B；尾轴对齐 C-010：fp32 计算域下 BLOCK_K/BLOCK_D 恒为 32B 倍数 |
| ⑤ | dtype 陷阱 | `dtype_support.md` §2.1/§3；`hardware_platforms.md` §3 | ❶ **新发现：int32 偏移溢出**（§0.6-3，6/13 benchmark case）；❷ 950 regbase 向量指令无 bf16 [SRC RegbaseUtils.cpp isHIVMSupportedType]——load 后立即 cast fp32（与旧版设计一致，现有源码背书）；❸ mask 保持 int32 比较（C-002）；❹ hIn 取列的 i32 **EQ** 比较向量安全（仅 i32 LT/GT/LE/GE 标量化）[SRC VCmpOp impl]；❺ 禁 int64 进入向量计算 lane（i64 全标量化 [SRC]）——int64 只许出现在标量地址基址 |
| ⑥ | 契约违禁写法 | `backend_contracts.md` C-001/C-002/C-006/C-007/C-011/C-014 | 无 `%`（无取模需求）；无 num_stages/num_warps；**grid 取 `num_aicore`**（含 tl.dot = CV 融合，C-011/operator_class_patterns——旧版取 num_vectorcore，已修正 §0.6-2）；本 kernel 非 µs 级（K 至 131072 的重 matmul），autotune 可保留但须守 C-014 三约束且交付前固化（C-007 精神） |

## 0.6 对旧版 P2 的修正（2 处错误 + 1 处潜伏 bug）

| # | 旧版论断 | 本版修正 | 依据 |
|---|---|---|---|
| 1 | §4「~110KB ≤ 192KB UB ✅」 | 950 按 **248KB 可用**校核（192KB 是 A2 值；结论方向不变、富余更大，n=32 大 tile 场景预算更宽裕） | hardware_platforms.md [SRC] |
| 2 | 骨架 `get_device_properties(...)["num_vectorcore"]` + §1「Mix 场景 grid 取物理核数即可，无需区分 AIC/AIV 角色」 | **CV 融合算子 grid = `num_aicore`**（Cube 数，Vector 按 1:2 协同）。「无需区分」是无实测等价断言，违反 C-012；按契约改为 num_aicore | C-011 [EXP+DOC]；operator_class_patterns.md [DOC] |
| 3 | 骨架寻址 `offs_m[:, None] * K`（int32 张量） | **潜伏 bug：T·K ≥ 2³¹ 时 int32 偏移溢出**——13 个 benchmark case 中 6 个触发（[16,16384,4,4096] 与 [8,16384,8,4096] T·K=4.3e9；[4,16384,8,4096]/[2,32768,8,4096]/[1,32768,16,4096]/[1,16384,32,4096] T·K=2³¹）。修正：**块基址用 int64 标量**（`x_base = pid 块起始行 × K`，标量不占向量 lane），块内偏移保持 int32（BLOCK_M·K ≤ 128×131072 = 1.7e7 安全）；阶段 3 的 `offs_m * (N·D)` 同法 | dtype_support.md §2.1 [SRC] i64 标量化规则 → int64 仅限标量地址基址；benchmark json 实测 shape |

（另：旧版骨架 `@triton.autotune` 四组 config 未含 num_stages/num_warps，合规；本版保留
autotune——本 kernel 非 µs 级，C-007 不适用，但 config 不得引入 num_stages/num_warps（C-006）。）

## 1. 分核策略 → grid 设计

| Ascend C 事实 | Triton 决策 | 可达性 | 依据 |
|---|---|---|---|
| 三模式路由：det3→BS；M≤10240 且 nD≥512 →MK；M≤256→ND；否则 BS（tiling.cpp:811-833） | **单一路径**：M 轴 BLOCK_M 分块，`NUM_BLOCKS = cdiv(T, BLOCK_M)`；K 轴不切核，program 内顺序循环 | ✅（结构性简化） | 旧版结论维持；split-K/atomic/两阶段全部是「多核协作 K 规约 + 保序」的实现手段，单 program 顺序 K 循环语义等价、天然确定、零 workspace |
| MK 模式 AIC 二维网格、K 部分和写 workspace、Part2 归并（tiling.cpp:689-766） | 不需要：K 循环在寄存器 acc 内累加，无中间落地 | ❌→语义等价替代 | 同上 |
| BS 模式 K=1024 顺序段 + SetAtomicAdd 保序（split_bs.h:198-208） | 不需要 | ❌→语义等价替代 | 同上 |
| ND 模式 K 按 AIC 均分 + AIV1 UB 归并（split_nd.h:117-129, 473-510） | 不需要 | ❌→语义等价替代 | 同上 |
| MK 两阶段 launch 内 pipe.Destroy 重建（apt.cpp:44-54） | 不需要 | ❌→语义等价替代 | 同上 |
| grid 设定：SetBlockDim(usedCoreNum=AIC 核数)（tiling.cpp:929；硬件拓扑 AIV=2×AIC 强校验 L575-579） | grid = **`num_aicore`** + persistent-core `for blk in range(pid, NUM_BLOCKS, NUM_CORE)` | ✅ | **C-011**：CV 融合算子 grid=Cube 数 [DOC]；operator_class_patterns §CV 融合。⚠️ 旧版取 num_vectorcore 已修正（§0.6-2）。核数运行时查询、禁止写死（hardware_platforms §3：950 AIC 4~36 跨度 9 倍） |
| 65535 逻辑核上限 | 不涉及：grid=物理核数 ≤ 36 | ✅ | C-004 |

## 2. tilingdata → BLOCK 超参

| tilingdata / 常量 | Triton 对应 | 候选值 / 推导 |
|---|---|---|
| totalLength=M / nD=K / fusionSize=matN | kernel 标量参 `T, K, MAT_N` | T = shape[0]（3 维）或 B·S（4 维）；K = n·D；MAT_N = 2n 或 n²+2n |
| chunkTSize（BS ≤128 / MK mL1Size ≤256） | `BLOCK_M` | {16, 32, 64, 128}，autotune key=["T","K","N"]（n 大时须缩小，见 §4 预算） |
| kUbSize/kL1Size（256~1024） | `BLOCK_K` | {64, 128, 256} |
| fusionAlign=RoundUp(matN,8) | 三段 tile 宽 constexpr | `BLOCK_NN = next_pow2(n)`（≥16，dot 最小维）；`BLOCK_RES = next_pow2(n²)`（n=32 → 1024）；列 mask `offs < n` / `offs < n²` |
| N=n | `N: tl.constexpr` | benchmark n ∈ {4,8,12,16,24,32}（**不硬编码 Ascend C 的 n≤8 约束**，旧卡 §9 已示警） |
| v1ChunkDSize=5120 | `BLOCK_D`（hIn 阶段 D 轴块） | {64, 128, 256} |
| normEps/hcEps/scaleMean | kernel 标量参 | scaleMean=1.0/K 由 kernel 内算 |
| outFlag/hasGamma/hasResi/implMode | `OUT_FLAG/HAS_GAMMA/HAS_RESI: tl.constexpr` + `DOT_PRECISION: tl.constexpr` | 编译期裁剪；DOT_PRECISION 0→"ieee"、1→"hf32" ⚠️[UNVERIFIED] |
| mkUseGmStage / workspace offsets / stage2* / cubeBlockDim* / multCoreSplitKSize | 无对应 | ❌ 全部消解 |
| deterministic level 3 | 无需区分：单 K 循环即确定 | 注释保留语义 |

## 3. tilingkey → constexpr

六实例 `(tilingMode∈{BS,ND,MK}, hasResi∈{0,1})` → **单 kernel + `HAS_RESI: tl.constexpr`**；
tilingMode 三态整体删除（分核统一后无分支语义）。HAS_RESI 控制 res 段 dot/store 与 hPost
的 hc_eps 项。op_impl_mode → `DOT_PRECISION` 传 tl.dot input_precision。

## 4. 片上内存 → 预算校核（按 950 = 248KB [SRC]）

| 成分（BLOCK_M=64, BLOCK_K=128, n≤8, fp32 计） | 字节 |
|---|---|
| x tile（load bf16 64×128×2B → cast fp32） | 16KB → 32KB |
| phi 三段 tile [128,16]×2 + [128,64] fp32 | 8+8+32 = 48KB |
| acc_pre/acc_post [64,16] + acc_res [64,64] fp32 | 4+4+16 = 24KB |
| sumsq [64] + alpha/bias 向量 | <1KB |
| hIn 阶段 x 重读 [64, BLOCK_D=64] bf16 ×n 头循环 + hin out（与上阶段复用分配） | ~32KB |
| **峰值合计（n≤8）** | **~110KB ≤ 248KB** ✅ 富余 2.3 倍 |
| **n=32 校核**（BLOCK_RES=1024，BLOCK_M=16，BLOCK_K=64） | x 16KB→32KB（cast 后释放 bf16）+ phi_res 64×1024×4B=256KB ⚠️ **超预算** → phi_res 须按 BLOCK_K 分列内层再切（或 BLOCK_K=32：128KB，加 acc_res 16×1024×4B=64KB + 其余 ≈ 210KB ✅ 临界）。**n=32 时 BLOCK_K≤32 且 BLOCK_M≤16**，写进卡片 Agent actions |

- Ascend C 三模式都需 GM workspace 中转（X staging / matmul partial / hMix，最大 M·matN·4B）；
  Triton 单 kernel 全程寄存器/UB，**workspace = 0**（out_flag=0 时）。
- 双缓冲由 multibuffer 承接（bishengir 默认 true [SRC Options.td]），**不配 num_stages**（C-006）。

## 5. 流水编排 → Triton 可达手段

| Ascend C 手段 | 可达性 | Triton 写法 / 依据 |
|---|---|---|
| X/gamma 预取下一片与当前片计算重叠（m_k_split.h:703-714） | ⚠️ 编译器自动 | K 循环天然形成 load/dot 流水 + multibuffer 默认开 [SRC]；无 num_stages（C-006） |
| L1/L0 乒乓、SetFlag/WaitFlag（cube_compute.h:97-162） | ❌ | 丢弃，tl.dot（T2 Cube）编译器管理 |
| 跨核 flag X_READY/X_CONSUMED/MM_READY + subblock 协议（common.h:95-99） | ❌ | 丢弃（无跨核协作）——同步开销消除 |
| Nd2Nz / Fixpipe 布局转换（cube_compute.h:59-134, 268-309） | ❌ | tl.dot 编译器管理 |
| AIV0/AIV1 subblock 对半分行（split_bs.h:148-163） | ❌ | 编译器内部 CV 协同（enable_auto_bind_sub_block 默认 true [SRC]） |
| hIn 阶段行批 2/4 + D 分块 5120（vector_compute.h:739-791） | ✅ | `for off_d in range(0, D, BLOCK_D)` + mask |
| CV 同步编排 | ⚠️ 编译器自动 | sync_solver 默认 true [SRC]；**不要手写 al.sync_block_***（extension_apis 纪律） |

## 6. Compute → Triton 计算骨架

| Ascend C 步骤（锚点，2026-09-14 复验） | Triton 表达 | 可达性 |
|---|---|---|
| x bf16→fp32 cast（vector_compute.h:396） | `tl.load(...).to(tl.float32)`（950 regbase 无 bf16 向量指令 [SRC]，立即 cast 是必须而非选择） | ✅ |
| xGamma = x·gamma（L397-401） | `x * g[None, :]`（HAS_GAMMA constexpr 裁剪） | ✅ |
| Σx² 部分和（L406-408） | `sumsq += tl.sum(x * x, axis=1)`（fp32 vreduce 向量安全 [SRC]） | ✅ |
| invRms：Muls(1/K)→Adds(eps)→Sqrt→Div（L429-433 ✅复验） | `inv_rms = 1.0 / tl.sqrt(sumsq * (1.0/K) + norm_eps)` | ✅ |
| matmul xGamma@phi^T（cube_compute.h 全文件） | phi 按行拆三段（pre: 0..n / post: n..2n / res: 2n..2n+n²），同一 x tile 三次 `tl.dot`，acc_pre/acc_post [BLOCK_M,BLOCK_NN]、acc_res [BLOCK_M,BLOCK_RES] | ✅（tl.dot=T2 Cube） |
| 段切分 ×alpha+bias（L100-128 ✅复验：pre=mm+0/post=mm+n/res=mm+2n） | 每段独立 tile：`p_seg = acc_seg * inv_rms[:,None] * a_seg[None,:] + b_seg[None,:]`；alpha/bias wrapper 预展开为长度 matN | ✅ |
| sigmoid=1/(1+exp(-x))；hPre 恒 +hcEps；hPost 2σ(+eps 仅无 res)（L596-603, 640-643 ✅复验） | `1/(1+tl.exp(-p))`；fp32 VExp/VDiv 向量原生 [SRC] | ✅ |
| hRes 直出（L119-126） | `p_res` 直接 store（无门控） | ✅ |
| hIn_d=Σ_i hPre_i·x_i,d，fp32 累加，CAST_RINT 回 b16（L658-791） | D 循环 + `tl.static_range(BLOCK_NN)` 头展开；取列 `tl.sum(tl.where(offs_nn==i, hpre, 0), axis=1)`——i32 **EQ** 比较向量安全 [SRC VCmpOp]；末段 `.to(x.dtype)` RTNE | ✅ |
| out_flag 调试输出 | `if OUT_FLAG:` store inv_rms/hpre/hmix（按 [pre\|post\|res] 段布局拼回） | ✅ |
| **寻址（§0.6-3 修正）** | 块基址 int64 标量 + 块内 int32 偏移 | ✅ |

### 不可达项汇总

1. ❌ 全部跨核同步 / 两阶段 / workspace 中转 → 单 program 顺序 K 循环替代（语义等价且确定性强）
2. ❌ L1/L0/Fixpipe 显式管理 → tl.dot 编译器自动
3. ⚠️ HF32（impl_mode=1）→ `input_precision="hf32"` [UNVERIFIED]，默认 "ieee" 对齐 FP32 标杆
4. ⚠️ 双缓冲/预取/CV 同步 → multibuffer / auto_bind_sub_block / sync_solver 编译器默认开

## 7. 精度策略 → 标杆与验收

- 计算全 fp32；仅输入 load 后 cast up、hin store 前 cast down（RTNE ≈ CAST_RINT）。
- 标杆：benchmark torch golden（`npu_benchmark/mhc/1_ManifoldConstrainedHyperConnectionPre.py`，
  全程 fp32，bf16 RTNE 输出）；语义五条已与 golden 逐条核对（旧卡 §9）。
- 判定阈值（evaluator 默认）：fp32 输出 rtol=2⁻¹⁰/atol=2⁻¹⁶/max_abs≤1e-2；h_in（bf16）
  rtol=atol=2⁻⁶。输入 x~U(-5,5) 幅值大，K=131072 时 fp32 累加误差显著，依赖 max_abs 与
  matched_ratio 兜底；用 rtol/atol 比对，不做 bitwise（Ascend C mmKStep=32 保序 vs tl.dot
  编译器定序）。
- impl_mode=1（HF32）单独放宽（建议 rtol=1e-3）。

## 8. 显式丢弃 / 不涉及清单

| 项 | 处理 | 理由 |
|---|---|---|
| 三 tiling 模式 / 六模板实例 | 合并为单 kernel | §1 已论证 |
| 跨核 flag / atomic 保序 / workspace / 两阶段 pipe 重建 | 丢弃 | 单 program 顺序 K 循环语义等价 |
| L1/L0/Fixpipe/subblock 分行 | 丢弃 | tl.dot + 编译器自动 |
| 910b/910_93 高阶 API 两阶段、n! factorial | 丢弃（进卡片架构差异节） | 目标平台 950 |
| deterministic level 3 特判 | 不需要 | 天然确定 |

## P2 质量门自查

- extraction 全部事实有去向：转化 16 条、结构性简化 6 条、显式丢弃 3 类 ✅
- 六任务检索全部命中 backend_facts 首层（未下沉 zh/IR 原文——首层均已带锚点）✅
- 旧版 2 处错误论断 + 1 处潜伏 bug 已修正（§0.6）；无「等价/不必复刻」类无证据断言（C-012 合规）✅
- 骨架 preflight 12 项 0 真命中（记录见 triton_skeleton.py 头注）✅
