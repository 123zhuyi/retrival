---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: MhcPre（mHC 超连接前置：RMSNorm+投影矩阵乘+分段门控+加权求和）Ascend C → Triton 语义转化卡
description: mHC 架构前置融合算子：x[bf16/fp16] 经 RMSNorm(Σx²) + 可选 gamma，与 phi[fp32] 做瘦矩阵乘得 hMix[T,matN]（matN=2n 或 n²+2n，n∈{4,6,8}），按 [pPre|pPost|pRes] 三段 ×alpha+bias 后分别过 sigmoid 门控得 hPre/hPost/hRes，最后 hIn_d=ΣhPre_i·x_i,d 输出 Attention/MLP 输入。Ascend C 为 CV 融合（MIX_AIC_1_2）三 tiling 模式（BS/ND/MK）+ 跨核 flag + workspace 中转；Triton 转化的关键决策：单 kernel 单 program 顺序 K 循环消解全部跨核协作，phi 拆三段三次 tl.dot，分段门控在独立小 tile 上完成。
tags: [mhc, mhc_pre, rmsnorm, matmul, fused_ops, sigmoid_gate, segment_split, cv_fusion, bf16, fp16, fp32_accumulate, ascend950, mhc_pre]
confidence: inferred
status: draft
resource: src/mhc/mhc_pre/（zmm_source_retrieval 本地源码，ascend950 arch35 为转化主线）
created_at: "2026-09-03T00:00:00Z"
updated_at: "2026-09-03T00:00:00Z"
---

sources:
  - url: src/mhc/mhc_pre/
    role: primary

# MhcPre（mHC 前置融合算子）

mHC（超连接）架构前置算子：一次调用完成 RMSNorm + 混合矩阵投影（matmul）+ 三段 sigmoid 门控 + 输入加权求和，产出 hIn（Attention/MLP 输入）、hPost、hRes（README L14-46）。**CV 融合 MIX_AIC_1_2**（apt.cpp:33），Ascend C 按 shape 路由三种 tiling 模式并用跨核 flag + GM workspace 协作；这些机制在 Triton 中全部消解（见 §4/§7）——本卡是与 mhc_sinkhorn（AIV-only）对照的 **CV 融合 / 含 tl.dot** 样例。

## 1. 功能与公式（详设 §1.1）

- 功能（README L16）：计算 mHC hidden 层的 H^res/H^post 投影矩阵与层输入 h^in
- **实现公式**（README L22-46，已与 kernel 逐步核对一致）：

$$
invRms=(mean(xFlat^2)+normEps)^{-1/2},\quad xGamma=xFlat\odot gammaFlat\ (\gamma\ 可选)$$
$$hMix=xGamma@\phi^T,\quad w=hMix\odot invRms$$
$$p = w\odot alphaVec + biasVec,\quad 段布局\ [pPre(n)\,|\,pPost(n)\,|\,pRes(n^2)]$$
$$hPre=\sigma(pPre)+hcEps,\quad hPost=2\sigma(pPost)\,(+hcEps\ 仅当\ \alpha=[2]),\quad hRes=pRes\ (\alpha=[3])$$
$$hIn_d=\sum_{i=0}^{n-1}hPre_i\,x_{i,d}$$

- **必须保留的语义点**（与代码核对）：
  1. 段顺序是 **[pre | post | res]**（vector_compute.h:107-125）
  2. invRms 顺序：`sum(x²)·(1/nD) + normEps` 再开方取倒数（vector_compute.h:429-433）
  3. hPre **恒加** hcEps；hPost 仅当无残差支路（alpha=[2]）才加 hcEps（vector_compute.h:602, 640-643）
  4. hIn 用 **sigmoid 之后** 的 hPre，fp32 累加后 CAST_RINT 回 bf16/fp16（vector_compute.h:720-722）
  5. sigmoid 即标准 1/(1+e^-x)（vector_compute.h:596-603）
- 变量：x [B,S,n,D] 或 [T,n,D]（T=B·S）；phi [matN, nD]；n∈{4,6,8}；D∈[16,16384] 16 对齐。

## 2. 原型与约束（详设 §2 / §2.1）

### 输入输出（来源：`op_host/mhc_pre_def.cpp:23-97`）

| 参数名 | 入参类型 | 必选性 | dtype | shape | 语义 |
|---|---|---|---|---|---|
| x | 输入 | 必选 | BF16/FP16 | (B,S,n,D)/(T,n,D) | mHC 层输入 |
| phi | 输入 | 必选 | FP32 | (2n,nD)/(n²+2n,nD) | 投影矩阵 |
| alpha | 输入 | 必选 | FP32 | (3)/(2) | 缩放参数；[3]=含残差支路 |
| bias | 输入 | 必选 | FP32 | 同 phi 首维 | bias |
| gamma | 输入 | 可选 | FP32 | (n,D) | RMSNorm 缩放 |
| hin | 输出 | 必选 | 同 x | (B,S,D)/(T,D) | Attention/MLP 输入 |
| h_post | 输出 | 必选 | FP32 | (…,n) | h_post 矩阵 |
| h_res | 输出 | alpha=[3] 时必选 | FP32 | (…,n,n)；alpha=[2] 时 shape=[0] | 未 sinkhorn 的 h_res |
| inv_rms/h_mix/h_pre | 输出 | out_flag=1 时必填 | FP32 | (…)/(…,matN)/(…,n) | 调试输出 |

属性（def.cpp:94-97）：`out_flag` int 默认 0；`norm_eps` float 默认 1e-6；`hc_eps` float 默认 1e-6；`op_impl_mode` int 默认 0（0=Cube FP32，1=HF32，仅 950/V2 接口）。

### 原型约束

- x rank 仅 3/4；phi 2 维；alpha 1 维长度 ∈ {2,3}（infershape.cpp:107-127）
- **n ∈ {4,6,8}**（tiling.cpp:58, 480-490）；**D ∈ [16,16384] 且 16 对齐**（tiling.cpp:63-64, 433-441, 492-496）
- phi 首维 = matN：alpha=[2]→2n；alpha=[3]→n²+2n（950 强校验，tiling.cpp:500-511）
- ⚠️ **偏差**：infershape 还接受 n!+2n（factorial，infershape.cpp:146-163），但仅 910b/910_93 kernel 有对应分支（mhc_pre.cpp:59-74），**950 tiling 拒绝 n! 形态**——Triton 实现只需支持 n²+2n
- hin dtype 必须等于 x（tiling.cpp:291-296）；out_flag=1 时三个调试输出必须非空（tiling.cpp:598-606）
- aclnn 两段式：V1 `aclnnMhcPreGetWorkspaceSize(x, phi, alpha, bias, gammaOptional, normEps, hcEps, hIn, hPost, hRes, invRmsOptional, hMixOptional, hPreOptional, ...)`（docs/aclnnMhcPre.md L64-81）；V2 多 opImplMode

## 3. 规格约束（详设 §3）

| 类别 | 取值 | 锚点 |
|---|---|---|
| 支持 NPU | Ascend 950PR/950DT、Atlas A3、Atlas A2 | README L5-13；def.cpp L81-92 |
| dtype | x/hin: BF16,FP16；phi/alpha/bias/gamma/其余输出: FP32 | def.cpp L23-79 |
| format | ND | def.cpp 全表 |
| 确定性计算 | 支持 deterministic level 3（强制 BS 模式保序累加，tiling.cpp:813-816）；**Triton 单 K 循环天然确定** | |
| 交付形式 | aclnn V1/V2 + GEIR 图模式 | examples/、docs/ |
| tilingkey | (tilingMode∈{BS,ND,MK}, hasResi∈{0,1}) 六实例 | tiling.cpp:914-917；tiling_key.h:27-50 |
| workspace | 系统区固定 20MB + 用户区（X staging / matmul partial / hMix） | tiling.cpp:65, 678-787 |

### 架构差异

- **ascend950（arch35，转化主线）**：Basic API 手写 Cube（Nd2Nz→L1→L0 乒乓→Mmad→Fixpipe），三模式路由，op_impl_mode 支持 FP32/HF32
- **910b/910_93**：高阶 matmul API 两阶段 Stage1/Stage2（mhc_pre.cpp:44-89），支持 n! factorial，n=4、D≤100000、128 对齐、impl_mode 仅 0（README L174-177）

## 4. 分核策略（详设 §4.1.1 → Triton grid）

- Ascend C 事实：**三模式路由**（tiling.cpp:811-833）：det3→BS；M≤10240 且 nD≥512 且 M≤AIC×256 → MK（M×K 二维 AIC 网格 + workspace 部分和 + 两阶段归并）；M≤256 → ND（K 按核均分 + UB 归并）；否则 BS（T 均分 + K=1024 顺序段 atomic 保序）
- **Triton 决策（结构性简化，单一路径）**：
  - 任务轴：**仅 M（=T）轴**，BLOCK_M 分块；**K 轴不切核**——单 program 内顺序 K 循环，累加在寄存器 acc 完成
  - 任务块总量：`NUM_BLOCKS = ceil(T / BLOCK_M)`
  - 核分配范式：
    ```python
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)   # grid = 物理核数
    for blk in range(pid, NUM_BLOCKS, NUM_CORE):
        ...
    ```
  - 尾块统一 `offs_m < T` mask
- **理由**：Ascend C 的 split-K/atomic/两阶段/跨核 flag 全部是为「多核协作算 K 维规约 + 保序」服务的实现手段；单 program 顺序 K 循环语义等价、天然确定、零 workspace。K=nD 最大 131072，循环内 load/dot 流水由编译器掩盖

## 5. BLOCK / Tiling 参数（详设 §4.1.2 / §4.1.3）

### tilingdata → Triton 超参映射

| tilingdata 字段 | 语义（tiling.h:31-60） | Triton 对应 | 候选值/推导 |
|---|---|---|---|
| totalLength / nD / fusionSize | M / K / matN | 标量参 `T, K, MAT_N` | T=shape[0] 或 B·S；K=n·D；MAT_N=2n 或 n²+2n |
| chunkTSize（BS≤128）/ mL1Size（MK≤256） | 单核 M 块 | `BLOCK_M` | {16, 32, 64, 128}，autotune key=["T","K","N"] |
| kUbSize/kL1Size（256~1024） | K 片 | `BLOCK_K` | {64, 128, 256}（UB 预算见 §6） |
| fusionAlign=RoundUp(matN,8) | 输出对齐 | 三段 tile 宽 `BLOCK_NN=16`、`BLOCK_RES=64` | n≤8、n²≤64 padding，列 mask `offs<n`/`offs<n²` |
| v1ChunkDSize=5120 | hIn 阶段 D 块 | `BLOCK_D` | {64, 128, 256} |
| N / D | n、D | `N: tl.constexpr` + 标量 D | N 进 constexpr 特化段边界 |
| normEps/hcEps/scaleMean | 标量 | kernel 标量参 | scaleMean 可由 1.0/K 替代表达 |
| outFlag/hasGamma/hasResi | 开关 | `OUT_FLAG/HAS_GAMMA/HAS_RESI: tl.constexpr` | 编译期裁剪 |
| implMode | Cube 精度 | `DOT_PRECISION: tl.constexpr`（"ieee"/"hf32"） | 传 tl.dot input_precision |
| mkUseGmStage / mkWorkspace* / stage2* / cubeBlockDim* / multCoreSplitKSize | MK 模式机制 | **无对应（全部消解）** | — |

### tilingkey → constexpr

六实例 `(tilingMode, hasResi)` → **单 kernel + `HAS_RESI: tl.constexpr`**。tilingMode 三态无残留语义（分核统一）；HAS_RESI 控制 res 段 dot/store 与 hPost 的 hc_eps 项。

## 6. 片上内存预算（详设 §5）

| 成分（BLOCK_M=64, BLOCK_K=128, fp32 计） | 字节数 |
|---|---|
| x tile（bf16 load 64×128 → cast fp32） | 16KB → 32KB |
| phi 三段 tile（[128,16]×2 + [128,64]） | 48KB |
| acc_pre/acc_post [64,16] + acc_res [64,64] | 24KB |
| sumsq / alpha / bias 向量 | <1KB |
| hIn 阶段 x 重读 + hin out（BLOCK_D=64，与上阶段复用分配） | ~32KB |
| **峰值合计** | **~110KB ≤ 192KB UB** ✅ |

- Ascend C 三模式都需 GM workspace（X staging 每核 2 槽 / matmul partial [groupK,M,matN] / hMix [M,matN]）；**Triton workspace = 0**（out_flag=0 时）
- BLOCK_K=64 时峰值降至 ~80KB，autotune 兜底

## 7. 流水编排与 Triton 可达性（详设 §6）

### 流水结构（转化后）

```
for blk:                              # M 块
  for k0 in 0..K step BLOCK_K:        # K 循环（load x/phi → cast/γ/Σx² → 三次 dot）
  invRms → 分段门控 → store hPost/hRes（/hMix/hPre/invRms）
  for d0 in 0..D step BLOCK_D:        # hIn：8 头展开 fp32 累加 → cast 回 b16 → store
```

### 可达性判定表

| Ascend C 手段 | 锚点 | 可达性 | Triton 写法 / 降级方案 | 预期影响 |
|---|---|---|---|---|
| DataCopyPad + Reg::Cast b16→fp32 | vector_compute.h:396, 439-453 | ✅ | `tl.load(mask, other=0.0).to(tl.float32)` | 等价 |
| Σx² 寄存器累加 + 每行一次 Reduce | vector_compute.h:406-408；split_nd.h:322-324 | ✅ | `sumsq += tl.sum(x*x, axis=1)` | 等价 |
| Basic API Mmad（L1/L0 乒乓、K=32 保序、Fixpipe） | cube_compute.h:136-309 | ❌ | `tl.dot(xg, phi_seg, input_precision=...)`，布局/流水全交编译器 | 不低于原实现（dot 映射同一片 Cube 硬件） |
| 跨核 flag（X_READY/X_CONSUMED/MM_READY + subblock 协议） | common.h:95-99；split_bs.h:67-99 | ❌ | **不需要**：无跨核协作 | 消除同步开销 |
| split-K 部分和 + SetAtomicAdd 保序 / 两阶段归并 | split_bs.h:198-208；m_k_split.h:283-545 | ❌ | 单 program 顺序 K 循环 | 语义等价且天然确定 |
| X/gamma 预取下一片（软件流水） | m_k_split.h:703-714 | ⚠️ | K 循环 + `multibuffer=True` 编译开关 | 编译器决定掩盖质量 |
| hIn 行批 2/4 + D 分块 5120 | vector_compute.h:739-791 | ✅ | `for d0 in range(0, D, BLOCK_D)` + mask | 等价 |
| HF32 模式（impl_mode=1） | cube_compute.h:105-111 | ⚠️ | tl.dot `input_precision="hf32"` | 支持度需实测标定，默认先 "ieee" |
| CAST_RINT 回 b16 | vector_compute.h:36-37, 720-722 | ✅ | `.to(x.dtype)`（RTNE≈RINT） | 等价 |

## 8. Triton 骨架（生成用伪码）

完整骨架见 `output/mhc_pre/triton_skeleton.py`（可直接翻译为 @triton.jit）。核心结构：

```python
@triton.autotune(configs=[(BLOCK_M, BLOCK_K, BLOCK_D) 四组], key=["T","K","N"])
@triton.jit
def mhc_pre_kernel(..., N: tl.constexpr, HAS_GAMMA/HAS_RESI/OUT_FLAG: tl.constexpr,
                   DOT_PRECISION: tl.constexpr, BLOCK_M/K/D: tl.constexpr,
                   BLOCK_NN=16, BLOCK_RES=64):
    for blk in range(pid, NUM_BLOCKS, NUM_CORE):
        # 阶段1：K 循环 —— x load(bf16→fp32) → ×gamma → Σx² ；同一 x tile 三次 dot
        #         （phi 拆 pre/post/res 三段行区间，acc_pre/acc_post/acc_res）
        # invRms = 1/sqrt(sumsq/K + norm_eps)
        # 阶段2：p_seg = acc_seg*inv_rms*a_seg+b_seg；hpre=σ(p_pre)+hc_eps；
        #         hpost=2σ(p_post)(+eps 仅无 res)；pres 直出；store hPost/hRes(/调试输出)
        # 阶段3：D 循环 —— hin_acc = Σ_{i<8} hpre[:,i]·x[:,i,d]（fp32）→ cast 回 b16 store
```

wrapper 要点：T=shape[0]（3 维）或 B·S（4 维）；K=n·D；matN=2n 或 n²+2n；**alpha 预展开**为长度 matN 向量（[a0]×n+[a1]×n+[a2]×n²，对应 VFExpandAlpha）；断言 x contiguous + bf16/fp16、phi shape==(matN, K)、n∈{4,6,8}、D∈[16,16384] 16 对齐。

## 9. 精度标杆与验收（详设 §测试设计）

- **⚠️ 定义域超集**：benchmark（`ascendc-kernelgen-data/npu_benchmark/mhc/1_ManifoldConstrainedHyperConnectionPre.{py,json}`，NPUKernelBench）case 的 **n 到 32**（`phi=[1088,131072]`，n²+2n=1088）、T 到 131072、K=n·D 到 131072（D=4096），全部 alpha=[3]、x 为 bf16、**无 out_flag、6 个输出全量返回**。超出 Ascend C 950 约束（n∈{4,6,8}、D≤16384）——Triton 实现**不要硬编码 n 范围**：
  1. `BLOCK_NN = next_pow2(n)`（≤32）、res 段 `BLOCK_RES = next_pow2(n²)`（n=32 时 1024）需按 n 选择，§8 骨架的 16/64 仅覆盖 n≤8
  2. UB 预算随 BLOCK_RES 重算（n=32 时 acc_res 一行即 4KB，BLOCK_M 需相应缩小，如 16）
  3. n>8 时 hIn 的 8 头展开循环改为 `tl.static_range(BLOCK_NN)` 上限
- **输出接口以 benchmark golden 为准**（`1_...Pre.py` L18-20, 75）：返回 `(h_in, H_post, h_comb_before, inv_rms, H_mix, H_pre)`——hRes 实际命名 h_comb_before；H_mix 为 **未乘 invRms 的 matmul 原值**（golden L52-53, 73）。golden 语义与卡片 §1 五条逐条核对一致（invRms 在 gamma 前计算 L46；hIn 用 sigmoid 后 H_pre×原始 x.float() L65-66；bf16 RTNE 输出）
- **判定阈值**（evaluator 默认）：fp32 输出 rtol=2⁻¹⁰、atol=2⁻¹⁶、max_abs≤1e-2；h_in（bf16）rtol=atol=2⁻⁶；benchmark 输入 x~U(-5,5)（非高斯，幅值大，fp32 累加误差比 randn 场景更明显）
- 标杆实现：benchmark torch golden（精度）+ evaluator 实测参考耗时（性能）
- 典型用例 shape（benchmark json）：(1,16384,4,4096)/(1,32768,16,4096)/(1,16384,32,4096)/(2,32768,8,4096)/(8,16384,8,4096)，gamma 有无交替；另保留 Ascend C UT 形态（alpha=[2]、gamma 缺省、异常 shape）
- 边界用例：n ∈ {4,8,12,16,24,32}；K 非 BLOCK_K 整数倍（K=49152/65536/98304 时 BLOCK_K∈{64,128,256} 均整除，但通用实现需 mask）；T 非 BLOCK_M 整数倍
- 精度语义说明：golden 全程 fp32（x cast fp32 后 matmul，phi 保持 fp32）；累加顺序差异（Ascend C mmKStep=32 保序 vs tl.dot 编译器定序）→ 用 rtol/atol 比对，不做 bitwise；K 大（131072）时 fp32 累加误差显著，依赖 evaluator 的 max_abs≤1e-2 与 matched_ratio 兜底

## 10. Agent actions（生成指引）

1. 识别类别：**融合算子**（rmsnorm + matmul + 分段门控 + 加权求和），CV 融合语义但 Triton 单 kernel 即可承载；matmul 为瘦矩阵（N≤80）、K 可达 131072。
2. 分核：仅 M 轴 BLOCK_M 分块，grid=物理核数，§4 范式；**K 轴不切核、不用 split-K/atomic**。
3. BLOCK 候选 §5（BLOCK_M {16..128}、BLOCK_K {64..256}、BLOCK_D {64..256}），autotune key=["T","K","N"]；UB 预算 §6。
4. **必须保留的语义点**（§1 五条）：段顺序 [pre|post|res]；invRms 先乘 1/K 再加 eps 再 sqrt 取倒；hPre 恒加 hc_eps、hPost 仅无 res 加；hIn 用 sigmoid 后的 hPre 且 fp32 累加；sigmoid=1/(1+e^-x)。
5. **phi 拆三段三次 tl.dot**（pre/post/res 行区间），acc tile 宽 16/16/64 padding + 列 mask；不要做成单大 tile 再列重映射。
6. alpha/bias 在 wrapper 预展开为长度 matN（alpha: a0×n+a1×n+a2×n²）。
7. **不要尝试复现** Ascend C 的跨核 flag、workspace 部分和、两阶段 pipe 重建、atomic 保序——单 program 顺序 K 循环语义等价（§7 表）。
8. impl_mode=1 时用 `DOT_PRECISION="hf32"`，精度判定单独放宽；默认 "ieee"。
9. out_flag=1 时 hMix 按 [pre|post|res] 段布局拼回 [T, matN] 写出。
10. 验收按 §9 四条标准执行；K 大时注意 atol 标定。

## 11. 复现性声明

本卡由 Ascend C 源码静态分析转化（confidence: inferred），锚点指向 `src/mhc/mhc_pre/`。未经硬件实测的点：BLOCK_M/K/D 最优值、相对 aclnn 性能、fp32/bf16 误差标定值、Triton-Ascend 的 input_precision="hf32" 支持度。已标注偏差 1 处：infershape 接受 n!+2n 但 950 tiling 拒绝（本卡不含 factorial）。910b/910_93 路径（高阶 API 两阶段）仅作架构差异记录，未转化。验收标杆已对齐 NPUKernelBench（npu_benchmark/mhc/1_ManifoldConstrainedHyperConnectionPre），其 n 范围（≤32）为 Ascend C 约束超集，BLOCK 超参需按 §9 泛化。回放验证：未做（P4 待执行，报告将补至 `output/mhc_pre/validation_report.md`）。
