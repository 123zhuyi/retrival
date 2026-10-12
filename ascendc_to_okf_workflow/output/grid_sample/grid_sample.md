---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: GridSample（cv_01，2D/3D 采样）Ascend C → Triton 语义转化卡
description: 首个「源丰富 + P2.5 全流程正向」验证算子（2026-09-17）。gather 类 Vector 算子：坐标映射+三 padding+三插值；P2.5 四门全跑通——编译门拦 3 陷阱（rint/反射早退/i64 索引）、正确性门 48 组合全过（修复 3 个交叉语义偏差）、board 门量化 vs CANN 原生 0.01-0.03x（gather 结构性税，C-018 立项）。验收参照系 = CANN 原生 aclnn（perf.json），非 torch 拼接。
tags: [grid_sample, gather, vision, interpolate, bilinear, bicubic, nearest, reflection_padding, fp16, cv]
confidence: measured
status: draft
resource: gitcode.com/cann/ops-cv image/grid_sample（本地 1-spare/ops-cv）；golden 1-spare/ascendc-kernelgen-data-perf-baseline（br_perf_baseline）level3/12_grid_sample
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-09-17T00:00:00Z"
---

sources:
  - url: gitcode.com/cann/ops-cv image/grid_sample（1-spare/ops-cv 本地 sparse checkout）
    role: primary（CANN 官方 AscendC 实现）
  - url: ascendc-kernelgen-data br_perf_baseline npu_benchmark/level3/12_grid_sample.{py,json}
    role: golden + 套件
  - url: 同目录 12_grid_sample_perf.json
    role: 验收基线（CANN 原生 aclnn 实测）

# GridSample（12_grid_sample，level3）

> 提取产物链：`output/grid_sample/{manifest, extraction, mapping}.md` +
> `triton_skeleton.py` + `p25_report.md`（P2.5 四门全记录）。
> **本卡特殊地位**：验收参照系是 CANN 原生 kernel（perf.json aclnn 实测），
> 首次「Triton 生成 vs CANN 原生」直接对标——结论先行：**正确性可全过，
> 性能有结构性差距（gather 税，C-018），本卡的价值在语义完备 + 差距定量**。

## 1. 功能与公式

每个输出位置 (n,c,ho,wo)：grid 归一化坐标 (gx,gy)（[-1,1]，x 沿 W、y 沿 H）
→ 反归一化 → padding 越界处理 → 插值采样。

- **反归一化**（kernel :622-631 / docs 公式逐式对应）：
  ac=True：`x'=(gx+1)/2·(W−1)`；ac=False：`x'=(gx+1)·W/2 − 0.5`；
- **padding**：zeros=角点 int 越界则贡献 0（权重清零等价）；border=**tap 级
  夹取**（浮点不动）；reflection=见 §7 的 mode 分叉（bilinear 预折叠+tap 夹取，
  bicubic 原始权重+逐 tap 浮点折叠）；
- **插值**：bilinear 4 点（floor+四权重）；nearest=**rint 银行家舍入**
  （CANN CAST_RINT，nearest.h:1039——非 floor(x+0.5)！）；bicubic 16 点
  **A=−0.75**（CubicConvolution1: alph=2−0.75/beta=0.75−3——与 torch
  grid_sampler 一致，与 upsample_bicubic 的 −0.5 不同）；
- **3D**：trilinear 8 点 / nearest；坐标多一维 z（grid 末维=3）。

## 2. 原型与约束

x/grid/y 同 dtype（fp16/fp32/bf16，ND；def :108-120）。套件实际：
**fp16×43 / fp32×7，无 bf16**。属性：interpolationMode / paddingMode /
alignCorners。约束：grid 末维 2|3；inH·inW（3D 含 D）≤ INT32_MAX
（tiling.cpp:205-208,241-244）；每维 ≥1。

## 3. 规格约束（50-case 套件）

47 个 4D + 3 个 5D（小 shape）；**bilinear×46** / nearest×2 / bicubic×2；
zeros×35 / reflection×8 / border×7；ac=False×42 / True×8；**H_out==H_in 恒成立
（纯 warp，无分辨率变化）**；N 1-4、C 2-112、H/W 24-1120；**grid 全随机
U[-1,1]（golden `_random_grid`）——无访存局部性，任何滑窗/行复用策略无效**。

## 4. 分核策略（→ Triton grid）

gather 类 Vector 算子（operator_class_patterns §1）。本骨架：
`grid=(N·NC·NHW,)`，program = (n, c_blk, hw_blk)，tile [BC, BHW]——
**坐标计算每块一次、BC 通道复用**（CANN general 版逐 channel 重复 gather 的
结构优势点①）。CANN 侧为 persistent-core（needCoreNum_ 由 tiling 定）+
CAL_H_W_BLOCK=512 + CHANNEL_BLOCK=64——本套件 task 数充足（大 shape 千级
program），Vector 类 grid 基准即可（族 §0）。

## 5. BLOCK / Tiling 参数

| 参数 | 起点 | 依据 |
|---|---|---|
| BHW | 256（pow2 ≥16） | 输出点块；P2.5 board 门可扫 |
| BC | min(next_pow2(C), 32) | 坐标复用度 vs tile 体积 |
| MODE/PAD/AC/IS_3D | constexpr 分派 | 编译期消化 flag |

## 6. 片上内存预算

无常驻状态。acc [BC,BHW] fp32 ≤ 32×256×4 = 32KB + 坐标/权重若干 KB ≪ 192KB。
**UB 不是本算子的约束——gather 发射才是（C-018）**。

## 7. 流水编排与可达性

- **CANN 侧两个结构性开销在本骨架消解**：① NCHW 逐 c 标量 gather 循环
  （MTE2ForNCHW :360-373）→ 坐标复用 + 向量 gather；② 4 角点 GM atomicAdd
  累加（:491-516）→ 片上累加单次 store；
- **但 gather 本身是结构性瓶颈（C-018）**：triton-ascend 的离散 gather 走
  SIMT 模板逐元素发射（[SRC] HIVMVectorOps.td:1828-1859）——恒等索引即慢
  24x、全随机再 ×4.6（P2.5 微基准）；对 CANN 原生实测 **0.01-0.03x**；
- **突围主路线（2026-09-17 实测定案，目标 0.8x 原生）**：**NHWC 段连续
  gather**——wrapper 一次性 NCHW→NHWC 转置（permute+contiguous，27MB 约
  180μs 量级）→ kernel 内每点每角点按 **[BHW, C] 段连续 load**
  （addr = tap_base[:,None]·C + c_vec[None,:]，行内连续行间随机）。
  **实测 [EXP-skeleton]：段连续 gather（段长 64B fp16）102μs ≈ 纯连续 load
  的 81μs 量级（268GB/s），对比全随机标量 gather 8952μs——88 倍 gather 税
  基本消除**。与原生同构：CANN 滑窗快路径也只支持 NHWC
  （fp16_slide_window.h:1353「滑窗只支持 channel last 场景」）+ UB 内硬件
  Gather 指令（PointBilinearXInLocal :1051-1067）——**原生赢法 = 数据进 UB
  + 片上随机访问，段连续 gather 是该结构在 Triton 可达语法下的对应物**。
  风险两态：① C 小时段长短（C=16→32B、C=2→4B）段连续优势退化，小 C case
  可能保留 NCHW 路径（dispatch 两态）；② 转置成本计入 wrapper（大 case
  摊薄，小 case 占比高）；
- 突围备选（分桶+片上 gather）：片上 tl.gather 实测全速（74μs≈连续 81μs，
  SIMT 税只在 GM gather）——但分桶 scatter 仍吃 GM 税（估算 ~53ms/百万点），
  **优先级低于 NHWC 段连续路线**，仅在段连续退化场景（极小 C）考虑；
- **套件 grid 全随机，滑窗/局部性类优化全部无效**（原生滑窗在本套件
  NCHW 下也退化——它的窗口判定含 channelLast!=NHWC 一律退化，:1353）；
- CANN 变体体系（slide_window/fullLoad/nearest/bicubic/arch35 smit）的
  分派逻辑：仅 NHWC 小输入走 fullLoad（tiling.cpp:182-197），套件 NCHW
  不触发；fp16 → slide_window、fp32 → normal（kernel 分派表 :82-146）。

## 8. Triton 骨架

`output/grid_sample/triton_skeleton.py`（v1，P2.5 全门通过版）。
要点：`_rint` 手写银行家舍入（**libdevice.rint 编译崩溃**——P2.5 编译门实证，
TritonToLinalg unsupported control flow）；`_reflect` 无分支（**运行时标量
早退同崩溃**）；gather 索引 int32（int64 仅标量基址）；zeros 用角点 int
判界；reflection/bicubic 逐 tap 折叠。

## 9. 精度标杆与验收

- golden = `F.grid_sample` 直调（融合版）——NPU 上分派到 CANN GridSampler，
  即**参照实现=CANN 原生**；smoke 已 48 组合全过（fp16 atol 2e-2 / fp32 1e-4，
  实测 fp32 最大差 ≤3.6e-7）；
- **性能验收 = perf.json（aclnn 实测基线）**：generator Phase 1 的
  ref_perf_json 注入机制可直接用——speedup 分母是 CANN 原生，~1.0x 为追平；
- 预期判据（预注册，勿移动球门）：iter_0 ≥45/50；gen ≤2（P2.5 已前置消化
  正确性）；**geomean 预期 0.01-0.05x 量级（C-018 结构性税）**。
  **判读三段论（防误读——0.01-0.05x 是起点基线的预期，不是目标也不是上限）**：
  ① 显著低于区间 = 卡片/代码问题（须修）；② 区间内 = 后端 gather 税，
  正常，agent 不应在 gather 形态上调 BLOCK 烧预算；③ 显著高于区间 =
  访存重组（分桶+片上 gather）落地成功——**Phase 4 的真正目标是把结果
  推进到区间之上**；
- P2.5 ⚠️ 项更新：片上 gather 已实测全速（⚠️→✅，突围的条件成熟）；
  分桶 scatter 成本为估算值（~53ms/1.25M 点）待实测。

## 10. Agent actions

1. 取 `triton_skeleton.py` 为起点（勿从零写；48 组合已验证）。
2. 语义三命门：**nearest=rint（非 floor(x+0.5)）**；**bicubic A=−0.75**；
   **padding 实现按 mode 分叉**（bilinear 预折叠 vs bicubic 逐 tap 折叠，
   border 一律 tap 级夹取勿动浮点）。
3. **勿用 libdevice.rint / 运行时标量早退 / int64 索引**（编译门三个实证
   陷阱，全部会崩）。
4. **0.8x 攻关路线（Phase 4 主攻，2026-09-17 实测定案）**：
   ① wrapper NCHW→NHWC 转置（permute+contiguous，成本计入）；② kernel 改
   段连续 gather：四角点各自 [BHW, C] 段 load（段内 C 连续），权重合成与
   写出不变（连续 store）；③ 实测定案依据：段连续 gather 102μs ≈ 连续
   81μs（88 倍 gather 税消除，§7）。**预期量级：大 case 0.8x+ 可达甚至
   反超**（原生 NCHW 走兜底标量路径——滑窗仅 NHWC；我们的转置+段连续
   等价于把原生的 NHWC 快路径显式做进 wrapper）。两态项：小 C（段长
   <32B）退化——C≤8 case 保留 NCHW 路径 dispatch 或实测定夺；3D 同理
   （段=C 连续）。若路线受阻：回 §9 三段论，勿在标量 gather 形态上调
   BLOCK 烧预算。
5. preflight 全清单 + **#11（gather 检测，C-018）**。
6. 长尾组合（nearest/bicubic/reflection/5D/fp32）正确性必须全过——smoke
   矩阵已覆盖，回放时不可退。

## 11. 复现性声明

- **confidence: measured（skeleton 级）**：语义经 48 组合 NPU 实测
  （P2.5 ②，对 torch 原生=CANN 原生）；性能差距经 board 门实测（7 case
  代表点 0.003-0.03x）；gather 税经微基准 + IR 源码双锚（C-018）。
- **回放已闭环（2026-09-17，`op_12_12_grid_sample_20260917_1453_7647`）**：
  精度实质 50/50（verify 7/50 = CPU golden 缺陷 + 43 fp16 经 32ULP 探针
  vs aclnn 全过）；**0.2211x vs 原生（判读三段论③突围成立但未达 0.8x）**；
  Phase 4 按卡 §10-4 路线 NHWC 段连续 gather **×34.6** 复现；
  剩余差距 = 段 gather 发射地板（31GB/s vs 微基准 213GB/s，大 shape L2
  坍塌）。**平台最佳生成 0.3939ms/1.4035x（反超原生）证明地板可破**——
  下一层主攻：片上 gather 窗口化（ir_3 编译失败的 IR 取证为前置）。
  完整复盘：`generation_retrospective_20260917.md`。
- **50-case 端到端回放未跑**——本卡的端到端数字待 P4 确认轮；P2.5 的
  小 shape/微基准上下文外推到 50-case 受 C-012 纪律约束。
- P2.5 simulator 门对本 kernel 采集失败（DBI init，kernel 特异性——94 骨架
  同时刻复测正常），已按覆盖门禁规则记录；微基准+board 门替代。

## 维护记录

- 2026-09-17 v1 初版：首个「源丰富 + P2.5 全流程正向」提取。P0-P3 全链 +
  四道实测门。编译门拦 3 陷阱（rint/反射早退/i64 索引）；正确性门修复 3 个
  交叉语义偏差（nearest zeros int 判界 / bicubic border tap 夹取 / bicubic
  reflection 逐 tap 折叠）；board 门量化 gather 税（0.01-0.03x vs 原生）；
  IR 门锚定 SIMT 模板机制（C-018 立项）。**P2.5 首次正向运行的全部四门
  均有产出**。
- 2026-09-17 **片上 gather 实测**：连续 load+tl.gather 74μs ≈ 纯连续
  load 81μs——SIMT 税只在 GM gather。§9 增判读三段论（防 0.01-0.05x
  预期被误读为目标/上限）。
- 2026-09-17 **0.8x 攻关路线定案（段连续 gather 实测）**：[BHW,C] 段连续
  gather（段长 64B fp16）102μs ≈ 纯连续 load（88 倍税消除）——NHWC 转置
  + 段连续 gather 为 Phase 4 主攻（§7/§10-4）。原生机制取证：滑窗快路径
  NHWC-only（fp16_slide_window.h:1353）+ UB 硬件 Gather 指令（:1051-1067）。
  分桶路线降级为备选。
- 2026-09-17 **回放闭环**：0.2211x vs 原生（×36.9 from Phase 3 基线，
  NHWC 段连续 gather 主路线 ×34.6 复现）；精度实质 50/50（verify 口径
  事故→C-019 立法，CPU golden fp16 缺陷）；0.8x 未达，剩余差距=
  段 gather 发射地板+大 shape L2 坍塌（下一层候选：原生 aclnn 布局
  转置取证 / 片上 gather ir_3 编译失败的 IR 取证 / 小 C 退化路径）。
