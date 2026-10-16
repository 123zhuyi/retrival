---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.2"
title: 依赖无关输入预加载（load 提前于等待边）与带宽仲裁意识
description: CV 依赖链（C1→V1→C2）中，V1 的计算虽须等 C1 结果，但其独立输入（u/g 等）的读取可提前发起；推而广之，只要 Pingpong 中输入缓冲槽位空闲即可先读。AscendC Fwd_h 实测预加载 +10%~22%。另需具备带宽仲裁意识：中间结果同时写 L1 与 GM 时竞争带宽，Triton 不可显式控制，记录判别出口。
tags: [preload, load_hoist, pipeline_hiding, bandwidth_arbitration, cv_dependency, gdn]
confidence: draft
status: draft
created_at: "2026-08-27T00:00:00Z"
updated_at: "2026-09-14T00:00:00Z"
---

## 依赖无关输入预加载与带宽仲裁意识（来源：AscendC GDN 流水线优化节）

【优化维度】：计算｜流水掩盖（访存子层）
【泛化层级】：条件性（迁移条件：存在 C1→V1 式依赖边、且 V1 有与 C1 结果无关的独立输入；或循环迭代间输入可预知；失效边界：所有输入均依赖前级结果，或输入缓冲在 Pingpong 中尚未释放——预读会破坏正确性）

<原则：模块间有数据依赖 ≠ 其输入读取也有依赖。V1 等待 C1 结果期间，MTE2 可提前把 V1 的独立输入（u、g）搬进 UB。推广规则：不管模块间是否存在依赖，只要存输入的缓冲槽位在 Pingpong 中被标记为可用，即可开始读操作。>

**正例（Triton-Ascend 表达）**：

```python
for i in range(n_chunks):
    # 下一次迭代的独立输入提前 load（与当前 C 链无依赖）
    u_next = tl.load(U + offs_next)
    g_next = tl.load(G + offs_next)

    with dl.async_task(scope="cube"):
        s_new = tl.dot(...)          # C1/C2 链
    with dl.async_task(scope="vector"):
        v_out = f(s_new, u_cur, g_cur)   # V1/V2：输入已就位
    u_cur, g_cur = u_next, g_next
```

**收益（AscendC 实现语境）**：Fwd_h 上预加载 +10%~22%；叠加"晚读出"后合计 +26%~58%（950 系列更显著）。

**带宽仲裁意识（晚读出）**：

- 现象：Fwd_h 的 v_out 既是 C2 输入（UB→L1）又是算子输出（UB→GM），两通路竞争带宽，可能拖慢关键路径上的 Cube2。
- AscendC 解法：人为延迟 UB2GM 通路，带宽优先给 UB2L1。
- ⚠️ **Triton 可达性：不可显式表达**。Triton 下只能间接缓解——把非关键的 `tl.store`（GM 写出）推迟到计算链之后发射、减少中间结果的 GM 落盘次数。判别出口：流水图确认瓶颈为 UB2L1/UB2GM 带宽竞争且间接手段无效时，转 AscendC（该手段在 950 上贡献显著）。

**注意事项**：

- 预加载增加 live range，UB 占用随之上升，需与双缓冲预算一起核算。
- 循环内预读要求下一迭代地址可计算（仿射下标）；data-dependent 下标不可预读。
- 输入复用是同类思想：Fwd_o 中 C1 与 C3 共用 Q，保持 Q tile 存活避免重复加载（Triton 表达为复用已 load 的变量，勿重复 `tl.load`）。

**已知实例（负结果，同样重要）**：

- Gdn2Fwd（chunk 递推 + CV 融合，ascend910b1）：K2 chunk 循环内三个依赖就绪 load（仅依赖 cc）提前至循环开头一并发射，verify 50/50 全过但 geomean 22.0006 vs 基线 22.2468 **无提升**。IR 归因：K2 dot 链（wW@st→uT→{wM@uT, kd@uT}）关键路径在 Cube 侧，load 重排未改变关键路径，编译器本就有等价调度空间。失效边界：**关键路径在 Cube 侧的 CV kernel，load 提前发射不构成优化**；先以流水图/IR 确认瓶颈边在访存侧再动手（2026-09-14；证据：op_94_94_Gdn2Fwd_20260914_1219_8124/output/opt_iter_0/{log.md,optimized_perf_result.json.speedup_vs_torch}）。

- Gdn2Fwd（chunk 递推 + CV 融合，ascend910b1）：serial persistent 骨架 double-buffer/load 重排（点 11）**正向**，geomean 21.2027 → 29.7598（×1.404，verify 50/50）。与上条负结果并存：**同一优化点在 serial persistent 骨架（访存侧有流水气泡）有效，在 chunked K2（Cube 侧关键路径）无效——失效边界在骨架/瓶颈边，不在优化点本身**（2026-09-16；证据：op_94_Gdn2Fwd_20260916_1619_2077/output/opt_iter_0/{log.md,optimized_perf_result.json.speedup_vs_torch}）。

<!-- okf:related:start -->

# 相关

- 实践案例: [ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡](../../ops/chunk-gated-delta-rule-fwd-h.md) — 主题相关：共享 gdn
- 实践案例: [GDN 线性注意力三阶段优化案例（FLA 移植 → Triton 体系化 → AscendC 重写）](../../ops/gdn-chunked-linear-attention.md) — 主题相关：共享 gdn
- 实践案例: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](../../ops/gdn2-fwd-chunked-wy-dispatch.md) — Gdn2Fwd 点 11 负结果（Cube 侧关键路径）为失效边界已知实例
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](../memory-sram/exp_solve_tril_002.md) — 同属优化知识，主题相关
- 相关主题: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](../memory-sram/layout-bthd-to-bhtd.md) — 主题相关：共享 gdn
- 相关主题: [preload + extract_slice 消除循环内同一数据重复 HBM 读取](../memory-sram/preload-extract-slice-eliminate-reread.md) — 主题相关：共享 preload
- 相关主题: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../memory-sram/ub-fill-and-compute-hoist.md) — 主题相关：共享 gdn
- 相关主题: [负优化排除清单：微小矩阵禁 tl.dot、constexpr 展开优先、preload 按 grid 规模决策](../meta-skills/exp_solve_tril_006.md) — 同属优化知识，主题相关
- 相关主题: [流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段](../meta-skills/pipeline-diagram-driven-diagnosis.md) — 主题相关：共享 gdn
- 相关主题: [chunk 递推算子的依赖二分法与 Pingpong 下发调度](chunk-recurrence-pingpong-dispatch.md) — 主题相关：共享 cv_dependency, gdn
- 相关主题: [负载均衡分核：grid 对齐物理核数，核内循环替代 GPU 多维 grid](load-balance-grid-sizing.md) — 主题相关：共享 gdn

<!-- okf:related:end -->
