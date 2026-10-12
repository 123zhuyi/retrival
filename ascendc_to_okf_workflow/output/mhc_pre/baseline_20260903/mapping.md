# P2 语义转化 — mhc_pre（Ascend C 事实 → Triton 决策）

> 依据：`../ascendc_triton_semantic_mapping.md`。每条给出：Ascend C 事实（锚点）→ Triton 决策 → 可达性。
> **核心结论先行**：本算子语义 = RMSNorm(Σx²) + 一次瘦矩阵乘 [T,nD]×[nD,matN]（matN≤80）+ 分段门控后处理 + 小矩阵加权求和 hIn。Ascend C 用三种 tiling 模式 + 跨核 flag + workspace + 两段 launch 解决的「K 轴分核归并」问题，在 Triton 中由**单 program 顺序 K 循环**天然消解——每 program 独占若干行的完整 K 轴，无需 split-K、无需原子加、无需跨核同步，且累加顺序天然固定（等价于 deterministic 语义）。三套模式合并为**一个 kernel**。

## 1. 分核策略 → grid

| Ascend C 事实 | Triton 决策 | 可达性 |
|---|---|---|
| 三模式路由：det3→BS；M≤10240 且 nD≥512 →MK；M≤256→ND；否则 BS（tiling.cpp:811-833） | **单一路径**：M 轴 BLOCK_M 分块，`NUM_BLOCKS = cdiv(T, BLOCK_M)`；K 轴不切核，program 内顺序循环 | ✅（结构性简化） |
| MK 模式 AIC 二维网格 mDim×kDim、K 部分和写 workspace、Part2 归并（tiling.cpp:689-766） | 不需要：K 循环在寄存器 acc 内累加，无中间落地 | ❌→语义等价替代 |
| BS 模式 K=1024 顺序段 + SetAtomicAdd 保序（split_bs.h:198-208） | 不需要：单 program 顺序 K 循环即固定顺序，天然确定 | ❌→语义等价替代 |
| ND 模式 K 按 AIC 均分 + AIV1 UB 归并（split_nd.h:117-129, 473-510） | 同上 | ❌→语义等价替代 |
| MK 两阶段 launch 内 pipe.Destroy 重建（apt.cpp:44-54） | 不需要：无跨阶段 | ❌→语义等价替代 |
| grid 设定：SetBlockDim(usedCoreNum)（tiling.cpp:929） | grid = AIC 物理核数 + `for blk in range(pid, NUM_BLOCKS, NUM_CORE)` 固定范式 | ✅ |
| 注意：Mix 场景 Triton program 实际占用 AIV（编译器调度），grid 取物理核数即可；无需区分 AIC/AIV 角色 | — | — |

## 2. tilingdata → BLOCK 超参

| tilingdata / 常量 | Triton 对应 | 候选值 / 推导 |
|---|---|---|
| totalLength=M / nD=K / fusionSize=N_mat | kernel 标量参 `T, K, MAT_N` | T = shape[0]（3 维）或 B*S（4 维）；K = n*D；MAT_N = 2n 或 n²+2n |
| chunkTSize（BS ≤128 / MK mL1Size ≤256） | `BLOCK_M` | {16, 32, 64, 128}，autotune key=["T","K","N"] |
| kUbSize/kL1Size（256~1024） | `BLOCK_K` | {64, 128, 256}（UB 预算见 §4） |
| fusionAlign=RoundUp(matN,8) | 三段 tile 宽 constexpr | `BLOCK_NN=16`（n≤8，dot 最小维）、`BLOCK_RES=64`（n²≤64）；列 mask `offs < n` / `offs < n²` |
| N=n ∈ {4,6,8} | `N: tl.constexpr`（运行时值进 constexpr 特化） | 段边界 n/2n 用 N 计算；BLOCK_NN=8 padding 用于 hIn 的 i 循环 |
| v1ChunkDSize=5120 | `BLOCK_D`（hIn 阶段 D 轴块） | {64, 128, 256} |
| normEps/hcEps/scaleMean | kernel 标量参（scaleMean 由 wrapper 算 1.0/K 传入或直接 K 算） | — |
| outFlag/hasGamma/hasResi/implMode | `OUT_FLAG/HAS_GAMMA/HAS_RESI: tl.constexpr` + `DOT_PRECISION: tl.constexpr` | 编译期裁剪 |
| mkUseGmStage / workspace offsets / stage2* | 无对应 | ❌ 全部消解 |
| deterministic level 3 | 无需区分：单 K 循环即确定 | 注释保留语义 |

## 3. tilingkey → constexpr

`tilingkey=(tilingMode, hasResi)` 六实例（tiling_key.h:27-50）→ **单 kernel + `HAS_RESI: tl.constexpr`**；tilingMode 三态整体删除（分核策略统一后无分支语义）。HAS_RESI 控制：段 3 计算与 h_res store、hPost 是否加 hc_eps。op_impl_mode → `DOT_PRECISION: tl.constexpr`（0→"ieee"，1→"hf32"）传入 tl.dot 的 input_precision。

## 4. 片上内存 → 预算校核

| 成分（BLOCK_M=64, BLOCK_K=128, fp32） | 字节 |
|---|---|
| x tile（load bf16 64×128×2B → cast fp32） | 16KB → 32KB |
| phi 三段 tile [BLOCK_K,16]×2 + [BLOCK_K,64] fp32 | 8+8+32 = 48KB |
| acc_pre/acc_post [64,16] + acc_res [64,64] fp32 | 4+4+16 = 24KB |
| sumsq [BLOCK_M] + alpha/bias 向量 | <1KB |
| hIn 阶段 x 重读 [BLOCK_M, BLOCK_D=64] bf16×8 头循环 + hin out | ~32KB（与上阶段复用分配） |
| **峰值合计** | **~110KB ≤ 192KB UB** ✅（BLOCK_K=64 时降至 ~80KB，autotune 兜底） |

关键差异：Ascend C 三模式都需 GM workspace 中转（X staging / matmul partial / hMix，最大 M·matN·4B）；Triton 单 kernel 全程寄存器/UB，**workspace 需求为 0**（out_flag=0 时）。

## 5. 流水编排 → Triton 可达手段

| Ascend C 手段 | 可达性 | Triton 处理 |
|---|---|---|
| X/gamma 预取下一片与当前片计算重叠（m_k_split.h:703-714） | ⚠️ | K 循环天然形成 load/dot 流水；Triton-Ascend `multibuffer=True`；卡片写约束「K 循环需要 MTE2/Cube 掩盖」 |
| L1/L0 乒乓、SetFlag/WaitFlag(MTE1_M 等)（cube_compute.h:97-162） | ❌ | 丢弃，tl.dot 编译器管理 |
| 跨核 flag X_READY/X_CONSUMED/MM_READY + subblock 协议（common.h:95-99） | ❌ | 丢弃（无跨核协作） |
| Nd2Nz / Fixpipe 布局转换（cube_compute.h:59-134, 268-309） | ❌ | tl.dot 编译器管理 |
| AIV0/AIV1 subblock 对半分行（split_bs.h:148-163） | ❌ | 编译器内部向量并行 |
| hIn 阶段行批 2/4 + D 分块（vector_compute.h:739-791） | ✅ | `for off_d in range(0, D, BLOCK_D)` + mask |

## 6. Compute → Triton 计算骨架（核心转化）

| Ascend C 步骤（锚点） | Triton 表达 | 可达性 |
|---|---|---|
| x bf16→fp32 cast（vector_compute.h:396） | `x = tl.load(...).to(tl.float32)` | ✅ |
| xGamma = x·gamma（vector_compute.h:397-401） | `xg = tl.where(HAS_GAMMA, x * g, x)`；g = load gamma 展平 [K] | ✅ |
| Σx² 部分和（vector_compute.h:406-408） | `sumsq += tl.sum(x * x, axis=1)` | ✅ |
| invRms=1/sqrt(mean+eps)（vector_compute.h:429-433） | `inv_rms = 1.0 / tl.sqrt(sumsq / K + norm_eps)`（或 Muls scaleMean 等价） | ✅ |
| matmul xGamma@phi^T（cube_compute.h 全文件） | **phi 按行拆三段**（pre: 0..n / post: n..2n / res: 2n..2n+n²），同一 x tile 三次 `tl.dot(xg, phi_seg, input_precision=DOT_PRECISION)`，acc_pre/acc_post [BLOCK_M,16]、acc_res [BLOCK_M,64]（n≤8、n²≤64 padding，mask 列 <n / <n²）——避免大 tile 列区间重映射 | ✅ |
| 段切分+×alpha+bias（vector_compute.h:103-127） | 每段独立 tile：`p_seg = acc_seg * inv_rms[:,None] * a_seg[None,:] + b_seg[None,:]`（alpha/bias wrapper 侧预展开，alpha→长度 matN：a0×n+a1×n+a2×n²） | ✅ |
| sigmoid=1/(1+exp(-x))；hPre +hcEps；hPost 2σ(+eps 仅无 res)（vector_compute.h:582-656） | `hpre = 1/(1+tl.exp(-p_pre)) + hc_eps`；`hpost = 2/(1+tl.exp(-p_post)) (+hc_eps if not HAS_RESI)` | ✅ |
| hRes 直出（vector_compute.h:119-126, 821-834） | `p_res` 直接 store（无门控） | ✅ |
| hIn_d=Σ_i hPre_i·x_i,d，fp32 累加，CAST_RINT 回 b16（vector_compute.h:658-726） | D 循环：x 按 [T,n,D] 视角 load `[BLOCK_M, BLOCK_D]`（第 i 张头），`hin_acc += hpre_col_i[:,None] * xi`（`tl.static_range(8)` 展开，取列用 where+sum 或编译期 slice，i≥n 置零）；store 时 `.to(x.dtype)`（RTNE≈CAST_RINT） | ✅ |
| out_flag 调试输出 inv_rms/h_mix/h_pre | `if OUT_FLAG:` store inv_rms、hpre，以及 acc_pre/acc_post/acc_res 按 [pre\|post\|res] 段布局拼回 hMix [T, MAT_N] | ✅ |

### 不可达项汇总（写入卡片 §7 与 §10）

1. ❌ 全部跨核同步 / 两阶段 / workspace 中转 → 单 program 顺序 K 循环替代（**语义等价且确定性强**）
2. ❌ L1/L0/Fixpipe 显式管理 → tl.dot 编译器自动
3. ⚠️ HF32（impl_mode=1）→ tl.dot input_precision="hf32"（Triton-Ascend 支持度需实测标定，默认先 "ieee" 对齐 FP32 标杆）
4. ⚠️ 双缓冲/预取 → multibuffer 编译开关

## 7. 精度策略 → 验收

- 计算全 fp32；仅输入 load 后 cast up、hin store 前 cast down（RTNE）
- 标杆（P4 自建）：torch 参考 `x.float()` 全程 fp64 真值 + fp32 对照；逐段公式复现（rmsnorm→matmul→分段门控→加权求和）
- 判定建议：hIn（bf16 输出）rtol=1e-2/atol=1e-2 对标 bf16 标杆；hPost/hRes/hPre/hMix fp32 输出 rtol=1e-5/atol=1e-6（matmul K 维累加误差随 K 放大，K=10240 时可放宽至 atol=1e-4，回放标定）
- impl_mode=1（HF32）需单独放宽（Cube 降精度），建议 rtol=1e-3
- 累加顺序差异说明：Ascend C FP32 模式 mmKStep=32 分组保序 vs Triton tl.dot 内部累加顺序由编译器定——fp32 逐元比对用 rtol/atol 而非 bitwise

## P2 质量门

- extraction.md 全部事实有去向：转化 16 条、结构性简化 6 条（三模式/workspace/跨核同步/两阶段）、显式丢弃 3 条（L1L0 管理/subblock 分行/atomic 保序，均注明替代）✅
- 骨架见 `triton_skeleton.py` ✅
