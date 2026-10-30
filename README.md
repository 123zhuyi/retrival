# MhcSinkhorn 优化路线档案

本目录按**优化版本推进顺序**整理 `experiments/r6_prof_guided/` 归因过程的核心产物。
一切都是「解 tmpj 路线（sk4→sk16→join→BM→EXACT→截断→平衡树）」，以**版本**为轴归纳，
同一 case 跨版本的纵向对比可直接对照 `0_总览报告` 里的 REPORT.md / CEILING.md 表。

> 归档原则：**每个 case 的整个 OPPROF 目录原样复制**（含 `visualize_data.bin`、`dump/DeviceProf*.bin`、
> 全部 csv / txt / collect.log）。MindStudio 可视化时**直接导入本归档里的 OPPROF 目录**（含 dump/）
> 即可，无需回原路径。`trace.json`（411MB 仿真器产物）过大，仅 `7_v23_v25b` 报告提及原路径，未复制。

---

## 目录总览

| 目录 | 内容 | 讲的是哪一步 |
|------|------|-------------|
| `0_总览报告/` | 各阶段报告 .md（阶段2入门 → 阶段5 v4sk2 诊断 → r6 全轨迹 → 天花板） | 全局 |
| `1_阶段5_v4sk2诊断/` | v4sk2 在 7 个 case 的**完整 OPPROF**（bin/dump/csv 齐全） | 起点瓶颈确认（只读一个 case 会误判） |
| `2_阶段6_v9_skrow/` | v9 skrow 在 case4/10/11/12 的**完整 OPPROF** | **核心转折**：56→32 blocks，去 padding 空车道，mte2 带宽 4.25→60.5 GB/s |
| `3_v10_v11_join/` | v10 join16 + v11 join 全系在 7 个 case 的**完整 OPPROF** | 消除 MTE3（L0C→UB fixpipe 结构代价） |
| `4_v12_BM重扫/` | v12 在 7 个 case 的**完整 OPPROF** + sweeps | 激光式 confirm 每一个 case 是否真的命中 |
| `5_v14_EXACT/` | v14 EXACT 两个**完整 OPPROF**（含 MemoryUB） | 找到真正的墙：UB 向量端口读写饱和（168.9 vs 211 GB/s） |
| `6_版本代码/` | 8 个优化版本的 `optimized_code.py`（重命名便于对照） | 每版改动可 diff |
| `7_v23_v25b/` | v25b 的仿真器报告 + 上板报告 | v14 之后不再按 profiling 驱动、改按算法/IR 迭代 |
| `9_成绩/` | `canbench/eval_v*.log` 各版本实测成绩 | 每版在 eval 集上的真值 |

> 每份 OPPROF 目录内部结构统一：`visualize_data.bin`（MindStudio 主可视数据）、
> `dump/`（内含 `DeviceProf*.bin` + `duration.bin` + `op_basic_info.txt` + `pc_start_addr.txt` +
> `aicore_binary.o`）、`PipeUtilization.csv`、`OpBasicInfo.csv`、`L2Cache.csv`、
> `ResourceConflictRatio.csv`、`collect.log`。

---

## 一条主线：为什么 v14 之后退出「profiling 驱动」

`1`~`5` 阶段完全靠上板 profiling 找方向：
- 阶段⑤ v4sk2：读 7 个 case 的 PipeUtilization，确认所有 case 都是 **VEC bound**（vec_ratio>80%），瓶颈在向量运算；
- 阶段⑥ v9 skrow：把 56 block 的 padding 空车道砍掉（→32），mte2 活跃带宽暴涨 14×，向量时间 -44%；
- v10/v11 join：消除 MTE3 往返；
- v12 BM：逐 case 激光确认；
- v14 EXACT + MemoryUB：发现 UB **向量端口带宽**（168.9 GB/s ≈ 理论 211 的 80%）是下一堵墙。

此后**板测已给不出方向**：VEC_ratio 停在 ~90% 这条终止线（见 `0_总览报告` README.md 的「占比陷阱」——ratio 不变不代表没优化，绝对时间在下）。于是 v23（自适应截断）、v25b（平衡树）、v27 改用**算法 / IR / 归约树**的手段，不再按 profiling 迭代。

---

## 性能真值（cv 各版本，取自 eval_*.log / REPORT.md）

| 版本 | 关键改动 | 说明 |
|------|---------|------|
| v4sk2 | 基线最优 | 56 blocks |
| v9 skrow | 去空车道 56→32 | mte2 bw 4.25→60.5，vec 时间 -44% |
| v10 join16 | 单 warp 合入 reduce | 消 MTE3 |
| v11 join全系 | 全 case 套用 join | — |
| v12 BM | block size 重扫 | 逐 case 确认 |
| v14 EXACT | 消除运行时标量比较 | + MemoryUB 定位 UB 端口墙 |
| v23 自适应截断 | 按收敛提前停迭代 | profiling 看不见（改的是轮数不是占用率） |
| v25b 平衡树 | 归约树平衡 | — |
| v27 | 最终 | — |

具体几何平均加速比逐行见 `9_成绩/` 各 eval 日志的末尾汇总（以实测 record 为准，本表不做二手抄写）。

---

## 常见操作

- **看 CSV**：用表格工具 / `python -m csv` 打开任一 `PipeUtilization.csv`，按 `sub_block_id` 分组看 `aiv_vec_ratio` / `aiv_scalar_ratio` / `aiv_mte2_active_bw`。
- **MindStudio 可视化**：直接导入本归档内任意 `OPPROF_*/` **整个目录**（含 `dump/DeviceProf*.bin`），不必回原路径推导 bin 位置。
- **diff 版本**：`6_版本代码/` 下任意两版 `diff v9_skrow.py v14_EXACT_vfm2.py` 可直接看改了什么。