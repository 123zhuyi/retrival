# 生成实验复盘 — grid_sample（12_grid_sample，level3）

> 2026-09-17 ｜ 实验：`op_12_12_grid_sample_20260917_1453_7647`（910b 卡 2）
> 知识条件：**grid_sample.md v1 卡唯一知识源**（template 全禁用、KB 跳过）——
> 首个「源丰富 + P2.5 正向 + vs CANN 原生验收口径」的完整闭环。
> 特殊性：本次回放中途撞上 verify 口径缺陷（CPU golden fp16 缺陷），
> 经独立取证裁决后继续——**verify 事故分析是本复盘的第一部分**。

## 1. 顶线结果

| 维度 | 值 |
|---|---|
| **精度（实质）** | **50/50 全对**：verify 口径 7/50（全部 fp32 过）+ 43 fp16 经 32ULP 探针 vs NPU aclnn **43/43 全过**（maxdiff ≤0.000977） |
| **性能（vs CANN 原生 perf.json）** | **0.2211x geomean**（impl 均延 2.50ms vs 原生 0.5528ms） |
| Phase 3→4 | 0.0060 → 0.2211（**×36.9**；91.49ms→2.50ms） |
| gen / opt | 2 / 4 |
| target_speedup | 5.0 未达——**口径问题**：config 默认值对「vs CANN 原生」参照系无意义（超原生 5x 物理不可能），非任务失败 |

## 2. verify 口径事故（全过程记录）

1. iter_0 verify 报 6/50（35 例「Framework 有 NaN」+ 9 例精度）——疑似大面积代码错误；
2. **独立取证（本工作流侧，全部可复现）**：① 全 50 case 孤立复跑参照零 NaN；
   ② iter_0 代码按 verify 同顺序孤立复跑 50/50 全过；③ 微实验证明
   **CPU fp16 grid_sample 自身产 NaN/大偏差**（bilinear 49-136 NaN/54 万元素）；
3. 定案：verify.py「CPU golden 优先」+ torch CPU fp16 grid_sample 实现缺陷
   ——43 个 fp16 case 的 golden 全是坏参照；**用户裁决：以 vs aclnn 验收继续**
   （benchmark --verify_not_required + perf.json 注入）；
4. **立法**：backend_contracts **C-019**（环境红线 AUTOLOAD=0 + golden 设备口径 +
   「先孤立复跑再改代码」排查顺序 + 红线应内建 verifier skill 而非 agent 笔记）。
   iter_1 的「tl.floor→gather 地址链边界取址偏差」修复是在脏参照下做的——
   未造成回退（终版 50/50 实质正确），但**在坏 golden 下修代码的风险这次侥幸
   避开，下次未必**。

## 3. 预注册判据核对（卡 §9，提取时冻结）

| # | 判据 | 结果 | 判定 |
|---|---|---|---|
| 1 | iter_0 ≥45/50 | verify 口径 6/50 但**孤立复跑 50/50**——卡片的实质正确性达标，口径污染不计入 | ✅（实质） |
| 2 | gen ≤2 | 2 | ✅ |
| 3 | 零勿重试违反 | Phase 4 全程按卡 §10 路线与勿重试清单 | ✅ |
| 4 | geomean vs 原生基线 | 0.2211——**落于判读三段论③（显著高于 0.01-0.05 起点区间 = 突围部分成功）** | ✅（突围成立） |
| 5 | 0.8x 攻关目标 | **未达**（差 ~3.6x） | ❌（见 §4 剩余差距分解） |

## 4. Phase 4 轨迹与剩余差距分解

| 轮 | 动作 | 结果 |
|---|---|---|
| opt_iter_0 | **NHWC 段连续 gather（卡片 §10-4 主攻路线）** | **×34.6**（0.0060→0.21 量级）——路线实测定案 |
| opt_iter_1_ir_1 | 掩码消除 | +6.5%（采纳，终版） |
| ir_2 | tl.trans 移除 | UB 溢出失败（回退） |
| ir_3 | **gather_out_to_ub（硬件/片上 gather 方向）** | **编译失败**（bishengir pipeline error 无详细行）+ 128B 成对段净零 |
| opt_iter_2 | simulator 诊断 | 无对应优化点，收官 |

**剩余差距（0.2211 → 0.8x）的分组分布**：
- **C>32：0.138** / C≤32：0.230 / C≤8：0.440——大 C 大 shape 最差；
- 最小 shape 反超：case 7（bicubic C=4 小 shape）**1.698x**、case 12 bicubic 0.851x、
  case 5（96²）0.777x——小 shape 上我们已追平/反超原生（原生 bicubic 16-tap 重，
  我们片上累加轻）；
- 生成侧诊断：**MTE2 段 gather issue-bound**（~278cyc/64B 段 ≈ 98ns，占 Vector
  cycles 85%；有效带宽 31GB/s ≪ 段连续微基准 213GB/s）——**发射路径饱和而非
  带宽饱和，是普通接口段 gather 的结构地板**。

**差距机制推断（[DERIVED]，待证）**：大 shape 随机段 gather 的 L2 命中率随
输入尺寸坍塌（462² fp16 plane=427KB→L2 压力），而原生若走 aclnn 层 NCHW→NHWC
转置+滑窗（85 的 TransposeLastTwo 先例，其 slide_window 快路径 NHWC-only），
则原生在 UB 内享受硬件 Gather——我们的 GM 段 gather 在 DDR 随机段上吃亏。
**下一层突围候选**（按优先级）：① 查清原生 aclnn 是否内置布局转置（ops-cv
op_api 层，影响对 0.8x 可达性的判断）；② 片上 gather+行带窗口（ir_3 的编译
失败需 IR 门取证根因——bishengir pipeline error 无详细行）；③ C≤8 退化组
单独路径。

## 5. 对机制验证的意义

- **P2.5 全流程正向首验成立**：编译门/正确性门拦截的 6 个缺陷全部在提取期清零
  → 生成侧 iter_0 实质一次写对（孤立 50/50），gen 2 轮收敛；
- **卡片主攻路线 ×34.6 复现**：P2.5 微基准（段连续 102μs≈连续 81μs）预判的
  方向被 Phase 4 独立执行并拿到数量级收益——**[EXP-skeleton] 微基准对端到端
  方向的预测力首次验证**；
- **verify 口径是新一类系统性风险**：C-019 立法前，fp16 算子的 verify 全部
  裸奔；本次若没做孤立复跑，正确代码会被误判返工（B3 型风险在验收层的镜像）；
- **0.8x 未达的诚实归因**：不是路线错误（路线贡献了 ×34.6），是段 gather 的
  发射地板 + 大 shape L2 坍塌——需要下一层结构（片上 gather/窗口化）或后端
  vgather 向量路径。

## 6. 数据与产物索引

- 实验：`op_12_12_grid_sample_20260917_1453_7647/`（report.md 瓶颈定量节、
  opt_iter_1_ir_3/log.md 的 ir_3 编译失败记录）
- 卡与骨架：`output/grid_sample/`（grid_sample.md、triton_skeleton.py、
  p25_report.md）
- 参照基线：`1-spare/ascendc-kernelgen-data-perf-baseline/.../12_grid_sample_perf.json`
- 原生机制取证：ops-cv `image/grid_sample/op_kernel/grid_sampler_2d_fp16_slide_window.h`
  （滑窗 NHWC-only :1353、UB 硬件 Gather :1051-1067）


## 7. 平台最佳生成锚点（2026-09-17 用户提供，平台评估行）

| 项 | 值 |
|---|---|
| 参考延迟（CANN 原生 aclnn） | 0.55283 ms |
| **平台最佳生成代码延迟** | **0.3939 ms → 加速比 1.4035（反超原生 1.4x）** |
| 精度 | 50/50 ✅；目标 0.6x 超额达成；备注「有 asc 专家手写参照」 |
| **我们本次终版** | 2.50ms → 0.2211x——**与最佳生成差 6.3x** |

**判读修正**：① 「0.8x 可达性」不再是推断——平台最佳已实证 1.4x；
② 我们 Phase 4 收官的「段发射地板 = 结构极限」判断**过早**——对方必然突破
了它（最大嫌疑路径：片上 gather + 数据暂存窗口化，即我们实测全速（74μs）
但 ir_3 编译失败的那条路）；③ 下一轮主攻 = IR 门取证 ir_3 编译失败根因
（bishengir pipeline error 无详细行）+ 片上 gather 窗口化变体——
**若拿到对方的实现/报告，直接做跨谱系提取（94 剧本）**。
