---
schema_version: okf.v1
kind: operator
op_name: apply_rotary_pos_emb
source_family: cann/ops-transformer
title: ApplyRotaryPosEmb（旋转位置编码，q/k 双路融合）
description: 对 query/key 两路执行 RoPE 并原地更新。三种旋转模式（half/interleave/quarter）× 四种 layout × cos/sin 广播两态 × partial rope。AIV-only 纯向量 elementwise，无规约。Triton 侧单 kernel + ROTARY_MODE constexpr + strides 布局归一化可全覆盖。
tags: [rope, rotary, position_embedding, transformer, elementwise, aiv_only, vector, arch35, ascend950]
confidence: verified
status: validated
created_at: "2026-09-08T00:00:00Z"
updated_at: "2026-09-14T00:00:00Z"
---

# ApplyRotaryPosEmb（旋转位置编码，q/k 双路融合）

> 材料：`output/apply_rotary_pos_emb/{manifest,extraction,mapping}.md`（2026-09-14 新流程复验版；
> 2026-09-08 初版与 P4 回放报告在 `baseline_20260909/`）。
> 本卡外部论断只挂引用不复写（维护纪律 README:83）：后端契约 → `backend_facts/backend_contracts.md`；
> 平台常数 → `backend_facts/hardware_platforms.md`；dtype 陷阱 → `backend_facts/dtype_support.md`；
> 编译选项 → `backend_facts/compiler_options.yaml`；算子范式 → `backend_facts/operator_class_patterns.md`。

## 1. 功能与公式

对 query、key 两路执行旋转位置编码并原地更新（推理侧融合两路为一次调用提性能，README.md:18）。
统一式（fp32 计算域）：`out = x·cos + rotate(x)·sin`，q/k 独立同式：

| rotary_mode | rotate(x) | 配对结构 | dSplitCoef |
|---|---|---|---|
| half（默认） | cat(−x₂, x₁)，前后半场 | out₁=x₁c₁−x₂s₁，out₂=x₂c₂+x₁s₂（common.h:111-119） | 2 |
| interleave | 奇偶对 (x[2i],x[2i+1])→(−x[2i+1],x[2i]) | out[2i]=xe·ce−xo·se，out[2i+1]=xo·co+xe·so（common.h:270-280） | 1 |
| quarter | cat(−x₂,x₁,−x₄,x₃)，四等分 | 1↔2、3↔4 各自 half 式（common.h:193-209） | 4 |

- **partial rope**：cos/sin 末维 realDim ≤ D；RD<D 时仅前 RD 维旋转，[RD, D) 原样直通
  （tiling_base_arch35.cpp:319；搬运双段化 common.h:781-877）。
- **aclnn 分层**：V1 = V2 锁 rotary_mode="half"（op_api/aclnn_apply_rotary_pos_emb.cpp:27-29）。
- **kernel 类型**：AIV-only 纯向量（apt.cpp:31 AIC 直接 return）——算子归类 **Vector 类**
  （operator_class_patterns §0），无 Cube、无 CV 融合问题。

## 2. 原型与约束

- 输入 query/key/cos/sin（REQUIRED，AutoContiguous），输出 query/key（原地语义，
  图级证据 fusion_pass tensormove）；dtype 全集 {F16, BF16, F32}（310p/kirin 裁 BF16）。
- 属性：layout∈{1=BSND,2=SBND,3=BNSD,4=TND} 默认 1；rotary_mode 默认 "half"。
- **arch35（950）约束**：cos==sin shape；cosN==1；q/k/cos 的 S 一致；cosD≤qD；cosB∈{qB,1}；
  D≤1024；half/interleave 要 D%2==0，quarter 要 D%4==0（tiling_base_arch35.cpp:142-207）。
- **legacy（910b 等）约束**：layout 仅 BSND/TND；D 与 cosD ∈{64,128}；rotary 仅 half
  （tiling.cpp:179-219）——架构差异见 §3。
- 空 tensor：V2 文档称支持，arch35 实现拒绝（CheckShapeAllPositive）——**以代码为准**。

## 3. 规格约束

| 类别 | 取值 |
|---|---|
| 目标平台 | **ascend950（arch35 主线）**；legacy 覆盖 910b/910_93/310p/kirin |
| dtype | arch35：FP16/FP32/BF16；推理系列（310p/kirin）无 BF16 |
| 确定性 | 是（纯 elementwise，无规约/atomic；workspace 16MB 仅系统保留） |
| 交付 | aclnn V1/V2 + GEIR |

### 架构差异（arch35 vs legacy）

arch35：4 layout、D≤1024、三模式、Regbase tiling 模板（key 20010/20011/20020/20030，
优先级 10000→40000 顺次尝试）；legacy：2 layout、D∈{64,128}、half only、
key{1,3,4}、BF16 走 AB_CAST+fp32 计算。**Triton 版以 arch35 为目标，legacy 约束不复刻**。

## 4. 分核策略（→ Triton grid）

- 源码三模板分核轴（BS 打平 / B×S 二维）都是「(b,s) 组空间」的不同切法；组间完全独立。
- **Triton 决策**：(b,s) 组为最小任务单元（组内 n 行单 tile 展开），persistent-core
  `grid = min(BS, vector_core_num)`，核内连续段循环（相邻 bs 共享 cos 行，L2 友好）。
  契约依据 C-011（`backend_facts/backend_contracts.md`）；核数运行时查询、**禁止写死**
  （`backend_facts/hardware_platforms.md` §3：950 向量核 8~72）。
- **Q/K 两次 launch**：[EXP] 实测定局 融合 3.8252 < 拆分 4.2060（baseline validation_report §5-10）。
- ⚠️ 裸 BS grid（一组一 program）是负引导（C-011：0.69x 实测）；65535 上限与
  TRITON_ALL_BLOCKS_PARALLEL 风险见 C-004（本算子 BS 量级通常无需考虑）。

## 5. BLOCK / Tiling 参数

| tilingdata | Triton 超参 | 推导 |
|---|---|---|
| realDim / dSplitCoef | `RD` + `SEG` + `ROTARY_MODE: tl.constexpr` | SEG=RD//2（half/interleave）或 RD//4（quarter） |
| dAlign | `BLOCK_DH` | `next_pow2(SEG)`；fp32 域恒 32B 倍数（C-010 天然满足） |
| ubFactorN 系列 | `BLOCK_N` | `next_pow2(N)`（N≤128 主流；更大改内层循环） |
| cos 广播（BA/BAB key） | `scb` | CosB==1 传 0，广播退化为寻址 |
| isPartialRope | wrapper 分支 | `out=x.clone()` 保底 + kernel 只写 [0,RD) |

形状/stride 全部 constexpr 化（C-013，[EXP] +2.2%）。

## 6. 片上内存预算

每迭代活跃数据 < 32KB（BLOCK_N=8/RD=128/fp16 例 ≈ 16KB），**vs 950 UB 248KB 可用**
（`backend_facts/hardware_platforms.md` [SRC]；注意不是 A2 的 192KB，UT mock 196608B 非规格值）——
富余 ~8 倍，容量不是瓶颈。双缓冲由 `multibuffer` 承接（bishengir 默认 true，
`backend_facts/compiler_options.yaml` [SRC]），**不要配 num_stages**（C-006 运行时不生效）。

## 7. 流水编排与可达性

| Ascend C 手段 | Triton | 判定（依据） |
|---|---|---|
| 三段式 CopyIn→Compute→CopyOut | (b,s) 组循环 load→compute→store | ✅（tl.load/store = T1，`backend_facts/tl_op_lowering.md`） |
| LoopModeParams 多维跨步 DMA | strides 参数化寻址 | ✅ |
| TQue 双缓冲 ×4 | multibuffer 选项（默认开） | ⚠️ 编译器自动 |
| **cos/sin 组级载一次**（aba_and_ba.h:185-196） | **组循环不变量外提，显式复刻** | ✅ 必须（[EXP] +100%；C-012 禁止「cache 驻留等价」断言） |
| MicroAPI DeInterleave/Interleave | 连续 load + `tl.reshape(...,2)` + `tl.split`/`tl.interleave` | ✅ fp32 向量原生（`backend_facts/dtype_support.md` §2.1 [SRC]） |
| ~~strided-2 半宽 load/store~~ | **禁用** | ❌ [EXP] 静默错值 + masked store 相邻行污染 + grid=1 不复现 |
| RegTensor\<float\> 全程 | `.to(tl.float32)` + 末段 RTNE | ✅ 精度必需 |
| Q/K 跨界块 / workspace | 两次 launch / 不需要 | 丢弃 |

同步：本算子纯 elementwise 无跨核依赖，编译器自动同步（GraphSyncSolver 默认 true，
`backend_facts/auto_passes.md` §1），**不要手写 al.sync_block_***（extension_apis §3 纪律）。

## 8. Triton 骨架

完整可运行骨架：`output/apply_rotary_pos_emb/triton_skeleton.py`（197 行，自包含，
含 wrapper 与标杆）。结构一句话：单 kernel + `ROTARY_MODE` constexpr 三分支 +
strides 布局归一化 + (b,s) 组 persistent 外循环 + 组首 cos/sin 外提 + fp32 计算域。
（P5 入库规范：骨架是 bundle 资产随卡分发，禁止本地路径死引用——baseline 教训 §4-②。）

## 9. 精度标杆与验收

- golden：`out = x.float()*cos + rotate(x.float())*sin` 三模式参数化，末段单次 `.to(dtype)`；
  填充幅度 [-1,1]（官方 ST 口径 fp16_error=2⁻⁸；fp16 大量级填充会撞 evaluator max_abs cap [EXP]）。
- 验收：fp32 rtol/atol=1e-5；fp16/bf16 ≤1 ulp；**模式专项**（half 前后半场交叉 /
  interleave 奇偶位 / quarter 四段）；**partial rope 专项** [RD,D) bitwise 等于输入。
- benchmark：`ascendc-migration/level2/5_ApplyRotaryPosEmb/` 45 精确 case
  （3 模式×4 layout×3 dtype×GQA/MQA×cosB 广播×partial rope）。

## 10. Agent actions

1. 取 `triton_skeleton.py` 为起点（勿从零写）；按目标 shape 确认 N≤128 或改内层循环。
2. 生成前过 preflight（`backend_facts/backend_contracts.md` 8 项 + `dtype_support.md` 4a~4d）：
   禁 `%`（kernel 内）、无 num_stages/num_warps、无 autotune 残留、mask 整型比较、
   grid=min(BS, 核数)、无 strided-2 访存、无 int64。
3. 编译选项：纯向量组——multibuffer 默认开即可；µs kernel 禁 autotune（C-007）。
4. 改动「等价/不必复刻」类断言前必须有实测或源码论证（C-012）。

## 11. 复现性声明

- 2026-09-08/09 P4 三组双盲回放 45/45 通过，geomean 4.2060（直给卡组）vs 3.1749（无知识对照）；
  报告与实验目录索引见 `baseline_20260909/validation_report.md`。
- 2026-09-14 新流程重提取：P0/P1 锚点抽验 4/4 有效；P2 六任务检索全部命中 backend_facts 首层；
  骨架 preflight 12 项 0 命中；新旧差异分析见 `ab_comparison.md`。

## 维护记录

- 2026-09-08 初版（旧流程）；2026-09-09 P4 升 verified，12 项缺陷修订回流。
- 2026-09-14 新流程重提取：外部论断改为挂 backend_facts 引用（UB 预算更正为 950=248KB、
  num_stages→multibuffer、strided-2 由「⚠️ 变体」改判 ❌ 禁用）；骨架与结论不变
  （backend_facts 本就从本算子 P4 蒸馏，自洽性符合预期）。
