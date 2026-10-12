---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: MhcPre（mHC 超连接前置：RMSNorm+投影矩阵乘+分段门控+加权求和）Ascend C → Triton 语义转化卡
description: mHC 架构前置融合算子：x[bf16/fp16] 经 RMSNorm(Σx²) + 可选 gamma，与 phi[fp32] 做瘦矩阵乘得 hMix[T,matN]（matN=2n 或 n²+2n），按 [pPre|pPost|pRes] 三段 ×alpha+bias 后分别过 sigmoid 门控得 hPre/hPost/hRes，最后 hIn_d=ΣhPre_i·x_i,d 输出 Attention/MLP 输入。Ascend C 为 CV 融合（MIX_AIC_1_2）三 tiling 模式 + 跨核 flag + workspace 中转；Triton 转化的关键决策：单 kernel 单 program 顺序 K 循环消解全部跨核协作，phi 拆三段三次 tl.dot，分段门控在独立小 tile 上完成。
tags: [mhc, mhc_pre, rmsnorm, matmul, fused_ops, sigmoid_gate, segment_split, cv_fusion, bf16, fp16, fp32_accumulate, ascend950]
confidence: inferred
status: draft
resource: src/mhc/mhc_pre/（zmm_source_retrieval 本地源码，ascend950 arch35 为转化主线）
created_at: "2026-09-03T00:00:00Z"
updated_at: "2026-09-14T00:00:00Z"
---

sources:
  - url: src/mhc/mhc_pre/
    role: primary

# MhcPre（mHC 前置融合算子）

> 材料：`output/mhc_pre/{manifest,extraction,mapping}.md`（2026-09-14 新流程复验版；
> 2026-09-03 旧流程初版在 `baseline_20260903/`）。
> 本卡外部论断只挂引用不复写（维护纪律 README:83）：后端契约 → `backend_facts/backend_contracts.md`；
> 平台常数 → `backend_facts/hardware_platforms.md`；dtype 陷阱 → `backend_facts/dtype_support.md`；
> 编译选项 → `backend_facts/compiler_options.yaml`；算子范式 → `backend_facts/operator_class_patterns.md`。

mHC（超连接）架构前置算子：一次调用完成 RMSNorm + 混合矩阵投影（matmul）+ 三段 sigmoid
门控 + 输入加权求和，产出 hIn、hPost、hRes。**CV 融合 MIX_AIC_1_2**（apt.cpp:33，2026-09-14
复验），Ascend C 按 shape 路由三种 tiling 模式并用跨核 flag + GM workspace 协作；这些机制
在 Triton 中全部消解（§4/§7）——本卡是**含 tl.dot 的 CV 融合样例**（对照：apply_rotary_pos_emb
为纯 Vector 样例）。

## 1. 功能与公式

- 功能（README L16）：计算 mHC hidden 层的 H^res/H^post 投影矩阵与层输入 h^in
- **实现公式**（README L22-46，已与 kernel 逐步核对，2026-09-14 复验 5/5 锚点）：

$$
invRms=(mean(xFlat^2)+normEps)^{-1/2},\quad xGamma=xFlat\odot gammaFlat\ (\gamma\ 可选)$$
$$hMix=xGamma@\phi^T,\quad w=hMix\odot invRms$$
$$p = w\odot alphaVec + biasVec,\quad 段布局\ [pPre(n)\,|\,pPost(n)\,|\,pRes(n^2)]$$
$$hPre=\sigma(pPre)+hcEps,\quad hPost=2\sigma(pPost)\,(+hcEps\ 仅当\ \alpha=[2]),\quad hRes=pRes\ (\alpha=[3])$$
$$hIn_d=\sum_{i=0}^{n-1}hPre_i\,x_{i,d}$$

- **必须保留的语义点**（与代码核对）：
  1. 段顺序 **[pre | post | res]**（vector_compute.h:100-128：pre=mm+0、post=mm+n、res=mm+2n）
  2. invRms 顺序：`sum(x²)·(1/nD) + normEps` 再开方取倒数（vector_compute.h:429-433）
  3. hPre **恒加** hcEps；hPost 仅当无残差支路（alpha=[2]）才加（vector_compute.h:602, 640-643）
  4. hIn 用 **sigmoid 之后** 的 hPre，fp32 累加后 CAST_RINT 回 bf16/fp16（vector_compute.h:720-722）
  5. sigmoid = 1/(1+e^-x)（vector_compute.h:596-603，Neg→Exp→Adds(1)→Div 显式序列）
- 变量：x [B,S,n,D] 或 [T,n,D]（T=B·S）；phi [matN, nD]。

## 2. 原型与约束

### 输入输出（`op_host/mhc_pre_def.cpp:23-97`）

| 参数名 | 入参类型 | 必选性 | dtype | shape | 语义 |
|---|---|---|---|---|---|
| x | 输入 | 必选 | BF16/FP16 | (B,S,n,D)/(T,n,D) | mHC 层输入 |
| phi | 输入 | 必选 | FP32 | (2n,nD)/(n²+2n,nD) | 投影矩阵 |
| alpha | 输入 | 必选 | FP32 | (3)/(2) | [3]=含残差支路 |
| bias | 输入 | 必选 | FP32 | 同 phi 首维 | bias |
| gamma | 输入 | 可选 | FP32 | (n,D) | RMSNorm 缩放 |
| hin | 输出 | 必选 | 同 x | (B,S,D)/(T,D) | Attention/MLP 输入 |
| h_post | 输出 | 必选 | FP32 | (…,n) | h_post 矩阵 |
| h_res | 输出 | alpha=[3] 时必选 | FP32 | (…,n,n) | 未 sinkhorn 的 h_res |
| inv_rms/h_mix/h_pre | 输出 | out_flag=1 时必填 | FP32 | (…)/(…,matN)/(…,n) | 调试输出 |

属性（def.cpp:94-97）：`out_flag` 默认 0；`norm_eps`/`hc_eps` 默认 1e-6；`op_impl_mode`
默认 0（0=Cube FP32，1=HF32，仅 950/V2）。

### 原型约束

- x rank 仅 3/4；phi 2 维；alpha 1 维长度 ∈ {2,3}（infershape.cpp:107-127）
- **Ascend C 950 约束：n ∈ {4,6,8}、D ∈ [16,16384] 且 16 对齐**（tiling.cpp:58, 63-64, 480-496）
  ——⚠️ **benchmark 是其超集**（n 至 32、K=n·D 至 131072），Triton 实现**不要硬编码 n 范围**
- phi 首维 = matN：alpha=[2]→2n；alpha=[3]→n²+2n（950 强校验 tiling.cpp:500-511）
- ⚠️ **偏差**：infershape 还接受 n!+2n（factorial），但仅 910b/910_93 有对应分支，
  **950 tiling 拒绝**——Triton 只需支持 n²+2n
- hin dtype 必须等于 x；out_flag=1 时三个调试输出必须非空

## 3. 规格约束

| 类别 | 取值 |
|---|---|
| 目标平台 | **ascend950（arch35 转化主线）**；910b/910_93 仅作架构差异记录 |
| dtype | x/hin: BF16,FP16；phi/alpha/bias/gamma/其余输出: FP32 |
| 确定性 | 源码支持 deterministic level 3（强制 BS 保序）；**Triton 单 K 循环天然确定** |
| 交付 | aclnn V1/V2 + GEIR |
| tilingkey | (tilingMode∈{BS,ND,MK}, hasResi∈{0,1}) 六实例 → Triton 单 kernel + HAS_RESI constexpr |
| workspace | 源码：系统 20MB + 用户区；**Triton = 0**（out_flag=0 时） |

### 架构差异

ascend950：Basic API 手写 Cube（Nd2Nz→L1→L0 乒乓→Mmad→Fixpipe），三模式路由，FP32/HF32
两态；910b/910_93：高阶 matmul API 两阶段，支持 n! factorial，n=4 only、impl_mode 仅 0。
**Triton 版以 950 为目标，老平台路径不复刻。**

## 4. 分核策略（→ Triton grid）

- Ascend C 三模式（BS/ND/MK）全部是为「多核协作算 K 维规约 + 保序」服务的实现手段。
- **Triton 决策（结构性简化）**：任务轴仅 M（=T），BLOCK_M 分块；**K 轴不切核**——
  单 program 顺序 K 循环，累加在寄存器 acc 完成，语义等价、天然确定、零 workspace。
- **grid = `num_aicore`**（本 kernel 含 tl.dot → **CV 融合算子**，契约 C-011 /
  `operator_class_patterns.md` [DOC]：CV 融合 grid 取 Cube 数，Vector 按 1:2 协同）；
  persistent-core 范式 `for blk in range(pid, NUM_BLOCKS, NUM_CORE)`；尾块 `offs_m < T` mask。
- 核数运行时查询（`get_device_properties`）、**禁止写死**
  （`hardware_platforms.md` §3：950 AIC 4~36 跨度 9 倍）。

## 5. BLOCK / Tiling 参数

| tilingdata | Triton 超参 | 推导 |
|---|---|---|
| totalLength / nD / fusionSize | `T, K, MAT_N` 标量参 | T=shape[0] 或 B·S；K=n·D；MAT_N=2n 或 n²+2n |
| chunkTSize / mL1Size | `BLOCK_M` | {16, 32, 64, 128}，autotune key=["T","K","N"] |
| kUbSize/kL1Size | `BLOCK_K` | {32, 64, 128, 256}（n=32 时 ≤32，§6） |
| fusionAlign | `BLOCK_NN` / `BLOCK_RES` constexpr | `max(16, next_pow2(n))` / `max(16, next_pow2(n²))`；列 mask `offs<n` / `offs<n²` |
| v1ChunkDSize=5120 | `BLOCK_D` | {64, 128, 256} |
| outFlag/hasGamma/hasResi/implMode | `OUT_FLAG/HAS_GAMMA/HAS_RESI/DOT_PRECISION: tl.constexpr` | 编译期裁剪；"ieee"/"hf32" ⚠️[UNVERIFIED] |

## 6. 片上内存预算

按 **950 UB = 248KB 可用**校核（`hardware_platforms.md` [SRC]；不是 A2 的 192KB）：

- **n≤8**（BLOCK_M=64, BLOCK_K=128）：x tile 32KB + phi 三段 48KB + acc 24KB + hIn ~32KB
  ≈ **110KB**，富余 2.3 倍 ✅
- **n=32**（BLOCK_RES=1024）：phi_res tile = BLOCK_K×1024×4B，acc_res = BLOCK_M×1024×4B
  ——**约束 BLOCK_K≤32 且 BLOCK_M≤16**（BLOCK_K=32/BLOCK_M=16 时 ≈210KB 临界 ✅；
  BLOCK_K=64 时 phi_res 一个 tile 即 256KB 超预算 ❌）。autotune config 已含 (16,32,64)。
- 双缓冲由 multibuffer 承接（bishengir 默认 true [SRC] `compiler_options.yaml`），
  **不配 num_stages**（C-006 运行时不生效）。

## 7. 流水编排与可达性

| Ascend C 手段 | Triton | 判定（依据） |
|---|---|---|
| CopyIn→Compute→CopyOut | M 块循环 + K 内循环 load→dot→store | ✅（tl.load/store = T1，`tl_op_lowering.md`） |
| Basic API Mmad（L1/L0 乒乓、Fixpipe） | `tl.dot(xg, phi_seg, input_precision=...)` | ✅（T2 Cube，编译器管理布局/流水） |
| 跨核 flag / split-K atomic 保序 / 两阶段归并 / workspace | **不需要**：单 program 顺序 K 循环 | ❌→语义等价替代（结构性论证，非 C-012 断言） |
| X/gamma 预取重叠（软件流水） | K 循环 + multibuffer 默认开 | ⚠️ 编译器自动 |
| CV 同步（AIV=2×AIC 协同） | sync_solver / auto_bind_sub_block 默认 true [SRC] | ⚠️ 编译器自动，**不手写 al.sync_block_*** |
| Σx² + invRms + sigmoid 门控 | fp32 向量原生（VAdd/VMul/VExp/VDiv 仅 i64 标量化 [SRC] `dtype_support.md`） | ✅ |
| hIn 取列 i32 EQ 比较 | 向量安全（仅 i32 不等值比较标量化 [SRC VCmpOp]） | ✅ |
| HF32（impl_mode=1） | `input_precision="hf32"` | ⚠️ [UNVERIFIED]，默认 "ieee" |

**❌ 不可达项：0 个**（三处 ❌ 均为「语义等价替代」而非能力缺失）。

## 8. Triton 骨架

完整骨架：`output/mhc_pre/triton_skeleton.py`（自包含，含 wrapper）。结构一句话：
单 kernel + M 轴 persistent-core 外循环 + K 轴顺序内循环三次 tl.dot（phi 拆 pre/post/res
三段）+ 分段门控独立小 tile + D 循环 hIn 加权求和。关键纪律：

- **int64 安全寻址**（mapping §0.6-3）：x 块基址用 int64 标量（T·K 可达 4.3e9 > int32，
  benchmark 13 case 中 6 个触发），块内偏移保持 int32；int64 只出现在标量地址基址，
  不进向量 lane（`dtype_support.md` §2.1）
- bf16 load 后立即 cast fp32（950 regbase 无 bf16 向量指令 [SRC]）
- mask 一律 int32 比较（C-002）；无 `%`、无 num_stages/num_warps（C-001/C-006）
- autotune 保留（非 µs 级 kernel，C-007 不适用）但守 C-014；交付前固化最优 config

## 9. 精度标杆与验收

- golden：benchmark torch 参考（`npu_benchmark/mhc/1_ManifoldConstrainedHyperConnectionPre.py`，
  全程 fp32，bf16 RTNE 输出）；返回接口以 golden 为准：`(h_in, H_post, h_comb_before,
  inv_rms, H_mix, H_pre)`——hRes 命名 h_comb_before；H_mix 为未乘 invRms 的 matmul 原值。
- 判定阈值（evaluator 默认）：fp32 输出 rtol=2⁻¹⁰/atol=2⁻¹⁶/max_abs≤1e-2；h_in（bf16）
  rtol=atol=2⁻⁶。输入 x~U(-5,5) 幅值大，K=131072 时 fp32 累加误差显著，依赖 max_abs 与
  matched_ratio 兜底；不做 bitwise（累加顺序差异：mmKStep=32 保序 vs tl.dot 编译器定序）。
- benchmark：13 case，shape 覆盖 (B,S,n,D) ∈ {1,2,4,8,16}×{16384,32768}×{4,8,12,16,24,32}×{4096}，
  全部 alpha=[3]、bf16、gamma 有无交替。**大 shape 触发 int32 溢出区**（§8），是正确性必测项。
- 边界补测：alpha=[2]（无 res 支路，hPost 加 hc_eps）、T 非 BLOCK_M 整数倍、K 非 BLOCK_K
  整数倍（通用 mask 已覆盖）。
- impl_mode=1（HF32）单独放宽（建议 rtol=1e-3）。

## 10. Agent actions

1. 取 `triton_skeleton.py` 为起点（勿从零写）。识别类别：**含 tl.dot 的 CV 融合算子**——
   grid 取 `num_aicore`（不是 num_vectorcore），编译选项按 `compiler_options.yaml`
   by_operator_class.cv_fusion（默认值均已最优，不需手设）。
2. **先核寻址安全**：T·K ≥ 2³¹ 的 shape 必须走 int64 标量基址路径（骨架已实现，勿回退
   为 `offs_m[:, None] * K` 纯 int32 写法）。
3. 按 n 选 BLOCK：n≤8 用默认 config；**n≥24 强制 BLOCK_K≤32、BLOCK_M≤16**（UB 预算 §6）。
4. 必须保留的语义五条（§1）：段顺序 [pre|post|res]；invRms 先乘 1/K 再加 eps 再 sqrt 取倒；
   hPre 恒加 hc_eps、hPost 仅无 res 加；hIn 用 sigmoid 后的 hPre 且 fp32 累加；
   sigmoid=1/(1+e^-x)。
5. phi 拆三段三次 tl.dot；alpha/bias wrapper 预展开为长度 matN。
6. 生成前过 preflight（`backend_contracts.md` 8 项 + `dtype_support.md` 4a~4d）：
   禁 `%`（kernel 内）、无 num_stages/num_warps、mask 整型比较、grid=物理核数（aicore）、
   无 int64 向量 lane、无 bf16 向量算术。
7. 不要复现跨核 flag、workspace 部分和、两阶段 pipe 重建、atomic 保序（§7 语义等价替代）。
8. 验收按 §9；大 shape case 必测（int32 溢出回归）。

## 11. 复现性声明

本卡由 Ascend C 源码静态分析转化（confidence: inferred）。2026-09-14 新流程复验：P0/P1
锚点抽验 5/5 有效；P2 六任务检索全部命中 backend_facts 首层；骨架 preflight 12 项 0 真命中。
相对 2026-09-03 旧版（`baseline_20260903/`）的修正：① UB 校核 192KB→248KB（950 [SRC]）；
② grid 查询 num_vectorcore→num_aicore（CV 融合契约 C-011）；③ 修复 int32 寻址溢出潜伏
bug（6/13 benchmark case 触发）。未经硬件实测：BLOCK 最优值、相对 aclnn 性能、hf32 支持度、
误差标定值。回放验证：未做（P4 待执行，报告将补至 `output/mhc_pre/validation_report.md`）。

## 维护记录

- 2026-09-03 初版（旧流程，confidence: inferred）。
- 2026-09-14 新流程重提取：外部论断改挂 backend_facts 引用；修正 §0.6 三项（UB 常数、
  grid 核数类型、int32 寻址溢出）；骨架同步修订。本算子**不是** backend_facts 契约的来源
  算子（契约来自 add/fused_bias_leaky_relu/apply_rotary_pos_emb 的 P4），本次修正构成
  对 backend_facts 的独立命中验证。
