# Triton 算子性能分析报告：MhcSinkhorn v4sk2（性能最优版本）

> 按 `triton-profiling-analysis` SKILL §5 结构输出。
> 采集链路：上板 `msprof op`（PipeUtilization/ResourceConflictRatio/L2Cache）+ 仿真 `msprof op simulator`（复用既有 `_sk4` 数据包）+ IR（同代码状态提取）。
> 本报告只做分析，不改代码；优化落地交主 agent / `triton-latency-optimizer`。

## 1. 分析对象与环境

| 项 | 值 |
|---|---|
| 分析对象 | **v4sk2** —— `output1014/MhcSinkhorn` 下官方评测得分最高的算子版本 |
| 实现文件 | `canbench/mhc_sinkhorn_v4sk/cann_bench/mhc_sinkhorn.py`（md5 `110ab8534db7a9d21cccb716d28831be`，2255 行） |
| 官方成绩 | **综合得分 70.16**（编译 20 + 精度 30 + 性能 20.16），平均加速比 0.7796x，精度 20/20（`cann_final_eval_20261019_211127.json`） |
| 版本谱系 | v1 60.34 → v2(r3多路径) 61.76 → v5 69.12 → v6 69.27/69.28 → **v4sk2 70.16（最优）** |
| kernel 构成 | 9 个 `@triton.jit` kernel + 静态分类分派（`classify()`，10 类）；与 v5_cls4/v8 共有 kernel **字节级相同**（已逐 kernel diff 验证） |
| 典型 case | 官方 20 case 全量（逐 case 分派矩阵见 §3.0） |
| 器件 | Ascend950PR_9579（DAV_3510），56 vector core + 28 cube core，1650 MHz，UB 248KB/core |
| CANN / Triton | cann-9.1.0 / triton 3.2.0 / torch_npu 2.9.0.post8 |
| 分析目标 | 上板判 bound → （无 bound 路径）仿真判气泡 → （SCALAR bound 路径）IR 标量降级下钻 → 给优化点 |
| 当前轮次 | v4sk2 首轮全量 profiling（r1） |

### 3.0 逐 case 分派矩阵（classify() 静态推导 × 官方 op_times 实测互证，一致）

| case | shape | iter | kernel | 配置 | 官方耗时 | 官方加速比 |
|---|---|---|---|---|---|---|
| 1 | [1,4,4] | 20 | `_sinkhorn_scale` | bm=1, grid=1 | 4.96µs | 1.0806 |
| 2 | [64,4,4] | 20 | `_sinkhorn_scale` | bm=2, grid=32 | 5.88µs | 0.9898 |
| 3 | [512,4,4] | 20 | `_sk4` | BM=32, grid=16 | 11.60µs | 0.7302 |
| 4 | [1024,4,4] | 20 | `_sk4` | BM=32, grid=32 | 12.02µs | 0.8319 |
| 5 | [4096,4,4] | 20 | `_sk4` | BM=128, grid=32 | 16.47µs | 0.6072 |
| 6 | [16384,4,4] | 20 | `_sk4` | BM=128, grid=128 | 39.10µs | **0.4066** |
| 7 | [1024,2,2] | 20 | `_sk2` | BM=32, grid=32 | 10.46µs | 0.9560 |
| 8 | [1024,3,3] | 20 | `_sk3` | BM=32, grid=32 | 11.14µs | 0.8977 |
| 9 | [1024,6,6] | 20 | `_sk6` | BM=32, grid=32 | 19.29µs | 0.5184 |
| 10 | [1024,8,8] | 20 | `_sk8` | BM=32, grid=32 | 31.75µs | **0.3150** |
| 11 | [1024,12,12] | 20 | `_sinkhorn_scale_masked` | N=16(pad), bm=4, grid=256 | 80.98µs | **0.1235（最差）** |
| 12 | [1024,16,16] | 20 | `_sinkhorn_scale` | N=16, bm=4, grid=256 | 57.55µs | **0.2693** |
| 13 | [1024,4,4] | 1 | `_sinkhorn_scale` | bm=4, grid=256 | 5.07µs | 0.8225 |
| 14 | [1024,4,4] | 5 | `_sk4` | BM=32, grid=32 | 9.66µs | 0.5849 |
| 15 | [1024,4,4] | 40 | `_sk4` | BM=32, grid=32 | 14.89µs | 0.6716 |
| 16/17/19/20 | [1024,4,4] | 20 | `_sk4` | BM=32, grid=32 | ~12µs | 0.82~0.84 |
| 18 | [1,2,2] | 20 | `_sk2` | BM=1, grid=1 | 4.05µs | **2.4691（最好）** |

权重结论：`_sk4` 覆盖 10/20 case（基本盘，0.41~0.84）；`_sinkhorn_scale_masked`(case11) 与 `_sinkhorn_scale`(case12) 是绝对成绩最差的两个 case。

## 2. 数据采集清单

| 数据路 | 来源 | 产物路径 | 代码状态指纹 | 完整性 |
|---|---|---|---|---|
| 上板 ×7 | 本轮新采（triton-simulator board 模式口径：`msprof op --aic-metrics=PipeUtilization,ResourceConflictRatio,L2Cache`） | `profiling/v4sk2_analysis/board/case{4,6,7,9,10,11,12}_*/OPPROF_*/` | v4sk2 md5 `110ab853…`（驱动 `board_drv.py` 原样加载） | ✅ 每路 3 CSV + OpBasicInfo 全量 |
| 上板（`_sk4` BM 扫描） | 复用既有 `/tmp/mhc_sk4/BM{32,64,128}`（2026-10-20 采，case5 形状 B=4096） | `/tmp/mhc_sk4/BM*/OPPROF_*/PipeUtilization.csv` | `_sk4` kernel 体与 v4sk2 字节级相同 | ✅（仅 PipeUtilization，无冲突/L2 表） |
| 仿真（`_sk4`） | 复用既有 `profiling/prof_out__sk4/`（2026-10-20 采，B=1024 n=4 iter=20，BM=32，即 case4 形态） | `profiling/prof_out__sk4/OPPROF_20261020111011_ZCYBNRMBVAQNHLZI/` | `_sk4` kernel 体与 v4sk2 字节级相同 | ✅ complete（32 core × 2 CSV + trace.json 95.8MB + visualize_data.bin + dump 三件套） |
| IR（`_sk6`/`_sk8`） | 本轮新采（triton-latency-optimizer `run_and_extract.sh`，驱动 `ir_drv.py` 跑 case9+case10） | `profiling/v4sk2_analysis/ir_output/_sk{6,8}_{ttir,ttadapter,last_pass}.mlir` | 与性能数据同一代码状态（同一 v4sk2 文件） | ✅ 三层 IR 齐全 |
| 官方逐 case 性能 | cann-bench 正式评测 | `cann-bench/reports/cann_final_eval_20261019_211127.json` | v4sk2 本体 | ✅ 20/20 |
| ~~v1 fused 上板+仿真~~ | `profiling/board*/`、`prof_out__mhc_sinkhorn_fused*` | 旧代码状态（v1 sha `ab7d2433…`）；且 `_mhc_sinkhorn_fused` 在 v4sk2 分派中为**死代码**（K_FUSED 无 case 命中） | 不匹配 | ❌ 未采用 |
| ~~r3/r5/实验 board~~ | `experiments/**/board*` | 旧代码状态（r2/r3/v5/v6 变体 kernel） | 不匹配 | ❌ 未采用（仅作先验证据引用其实验结论） |

**数据缺口**：① `_sk4` BM128 配置（case6）无仿真包（仅 BM32）；② `_sk2`/`_sk3` 无仿真包；③ `_sinkhorn_scale` 小 bm 配置（case1/2/13）未单独上板（由 case12 同 kernel 数据旁证）；④ case8(`_sk3`) 未上板（0.8977 接近基线，非主攻方向）。

## 3. 上板 bound 判定

### 3.1 各 pipe 占比表（本轮新采，各核中位数；附带 Task Duration / 核数）

| case | kernel（配置） | Task Duration | **vec** | scalar | mte2 | mte3 | bound 结论 |
|---|---|---|---|---|---|---|---|
| 4 | `_sk4` BM32, grid=32 | 12.74µs | 42.28% | 25.54% | 21.84% | **38.68%** | **无 bound**（最高 MTE3 <50%） |
| 6 | `_sk4` BM128, grid=128 | 38.63µs | 43.28% | 23.69% | 35.04% | **51.51%** | ✅ **MTE3 Bound**（最高且 >50%） |
| 7 | `_sk2` BM32, grid=32 | 10.72µs | 29.39% | 17.21% | 22.95% | **41.77%** | 无 bound |
| 9 | `_sk6` BM32, grid=32 | 19.67µs | **51.76%** | **32.30%** | 15.98% | 31.49% | ✅ **VEC Bound** + SCALAR>30%（双命中） |
| 10 | `_sk8` BM32, grid=32 | 30.51µs | **52.04%** | **36.64%** | 13.84% | 30.37% | ✅ **VEC Bound** + SCALAR>30%（双命中） |
| 11 | `_sinkhorn_scale_masked` N=16/bm=4, grid=256 | 81.57µs | **94.01%** | 17.64% | 4.16% | 2.00% | ✅ **VEC Bound，>80% 理想终止线** |
| 12 | `_sinkhorn_scale` N=16/bm=4, grid=256 | 58.43µs | **97.14%** | 12.96% | 3.60% | 1.84% | ✅ **VEC Bound，>80% 理想终止线** |

复用数据交叉验证（`_sk4` @ B=4096 形状，`/tmp/mhc_sk4`）：BM32 vec 45.92%/mte3 **51.87%**、BM64 mte3 39.65%、BM128 mte3 38.27% —— 与本轮 case4/case6 同一量级、同一结构（MTE3 与 VEC 双高、无单边压倒），结论互洽。

### 3.2 附带问题（与 bound 并行核对）

| 问题 | 判定条件 | 实测 | 结论 |
|---|---|---|---|
| Bank/LDU 冲突 | `aiv_vec_ldu_cflt_ratio > 5%` | case12 **40.39%**、case11 **32.00%**、case6 **9.88%**；case10 4.47%、case9 3.84%、case4 1.67%、case7 1.08% | ✅ case11/12/6 **命中**（向量 load 单元冲突显著，与 reduce 树/gather 布局互证） |
| 核间负载不均衡 | 各核 `aiv_time` 差异 >10%（(max−min)/median） | case6 **49.8%**（24.1~38.0µs）、case7 25.5%、case11 **23.0%**、case12 **21.3%**、case4 16.7%、case9 11.8%、case10 8.1% | ✅ 6/7 命中；叠加 grid 结构看更本质：case4/7/9/10 **grid=32 只用 32/56 核（57% 核利用率）**；case6 grid=128 → 2.29 波尾波空转；case11/12 grid=256 → 4.57 波 |
| L2 命中率低 | `aiv_read_hit_rate < 50%` | case12 读 **0.62%**、case11 读 **10.89%**；其余 ≥96.9% | 形式上命中，但 case11/12 的 MTE2 占比仅 4.16%/3.60%（绝对访存量小）——与 v1 同款结论：**不独立立项** |
| STU 冲突 | `aiv_vec_stu_cflt_ratio > 5%` | 全部 ≤1.59% | 无 |

### 3.3 bound 总结论

- **两条 VEC Bound 已达 80% 理想终止线**（case11 94.01% / case12 97.14%）：向量 pipe 已近打满，
  继续优化只能靠**减少向量指令总量**（换算法结构/换 kernel 形态），调度层面无空间。
- **case6 MTE3 Bound**（51.51%）：`_sk4` 大 B 配置的写出 pipe 登顶，叠 9.88% LDU 冲突与 2.29 波尾波。
- **case9/10 VEC+SCALAR 双命中**：SCALAR 32.30%/36.64% >30% → 按 §3A.1 强制 IR 下钻（见 §4.5）。
- **case4/7 无 bound** → 仿真气泡路径（`_sk4` 有数据，见 §4；`_sk2` 记数据缺口）。

## 4. 气泡分析（无 bound 路径：`_sk4` case4 形态，仿真包复用）

> 工具：`pipe_overlap_analysis.py --dir`（32 核自适应聚焦 core0 两子核 + 跨核汇总）与
> `blocking_point_analysis.py --pairs`（core0.veccore0/trace.json）；原始结果
> `v4sk2_analysis/sk4_pipe_overlap.json`。

### 4.1 每 pipe 空泡表（core0 两子核中位数；span=9.97µs/子核）

| pipe | merged（队列占用） | wait | eff（有效） | 空泡率 | 成因 | 可优化 |
|---|---|---|---|---|---|---|
| **MTE3**（写出） | 59.0% | 11.2% | **48.1%** | 41.9%+11.2% 驻留 | **等 VEC 迭代链结果**（WAIT_FLAG 证据 ↓） | 依赖链固有，部分可前移 |
| **RVECSU**（向量执行） | 40.7% | 0 | **40.7%** | 59.3% | 期间 MTE3 活跃 78%：与写出相分离 | 受依赖链约束 |
| SCALARLDST | 22.2% | 0 | 22.2% | 77.8% | gather 地址生成，与 MTE3 相分离 | 连续化可减少 |
| MTE2（读入） | 15.1% | 0 | 15.1% | 85.1% | 仅开头一段集中搬运 | 连续化可减少 |
| SCALAR | 8.8% | 0 | 8.8% | 91.2% | 物理空闲为主 | — |
| VECTOR（同步队列） | 48.1% | 48.0% | 0 | — | 仅 WAIT_FLAG 驻留 | — |

### 4.2 掩盖失效对（重叠度 <10%）与阻塞点源码定位

| 失效对 | 重叠度 | 双方主导代码行（v4sk2 行号，与 v8 一致） |
|---|---|---|
| **MTE2 × MTE3** | **0.81%** | MTE2 侧：`Copy1D.cpp`/`DMAUtils.h`（⚠外部行，16 路 stride-gather load，v4sk2:147-162）；MTE3 侧：v4sk2:202-209 `tl.store(optr + base + k, oK, mask=ok)` |
| **MTE3 × RVECSU** | **1.88%** | MTE3：v4sk2:202-209 store；RVECSU：v4sk2:190-191 `rvK = rvK / (rvK * wcK + eps)`、:195 `wtr3 = w3*rv0 + …`（20 轮迭代链主体） |
| MTE3 × SCALAR | 4.13% | SCALAR 侧 `VecUtils.h:137`⚠外部 / internal（无用户行） |
| MTE3 × SCALARLDST | 5.37% | SCALARLDST：v4sk2:195 `wtr3`、:196 `cv0 = cv0 / (cv0 * wtr0 + eps)` |

**WAIT_FLAG 归因**：`MTE3 ← VEC` ×11 合计 **3.82µs（占 span 38.3%）** —— 写出 pipe 近四成时间在
等迭代链最后一次除法的结果；`MTE2 ← VEC` 0.97µs、`VEC ← MTE3` 1.12µs 次要。
周期性阻塞点（≥3 次均匀重复）未检出 —— 空泡是**单段式**的（首 load → 中段算 → 尾 store）。

### 4.3 气泡结论

`_sk4` 的空泡结构是**三段相分离**：16 路 stride-gather load（Copy1D 段式搬运，MTE2 15%）
→ 20 轮寄存器迭代链（RVECSU 41%）→ 16 路 stride store（MTE3 48%+11% 等待）。
单 program 内 load/compute/store 被数据依赖彻底串开（MTE2×MTE3 重叠 0.81%），
**程序内双流水无可开**；出路在**程序间并行度**——grid=32 只用 57% 的核，
提升 grid 让不同 program 的 load/compute/store 在核间互相掩盖。

## 4.5 IR 分析（SCALAR bound 强制下钻：`_sk6`/`_sk8`，case9/10）

> IR 产物：`v4sk2_analysis/ir_output/_sk{6,8}_{ttir,ttadapter,last_pass}.mlir`（与板测同代码状态）。
> 工具：`ir_loc_resolver.py <kernel>_ttir.mlir`（#loc 锚点 → 源文件:行:列 + 形态归类）。

**热行清单与降级形态**（ttir 层，脚本化归类）：

| kernel | 指针张量 gather | 指针构造（splat/addptr 前驱） | 逐元素比较/算术 | cumsum/标量归约 | 常量未标记 |
|---|---|---|---|---|---|
| `_sk6` | **72 ops**（v4sk2:344-379 load ×36 + 对应 store ×36） | 76 ops（同址） | 1（`:342 ok = off < B`，掩码必需） | 0 | 无（iter_step/BLOCK_M 均 constexpr） |
| `_sk8` | **128 ops**（v4sk2:626-689 load ×64 + 对应 store ×64） | 132 ops（同址） | 1（`:624 ok = off < B`） | 0 | 无 |

- **降级形态判定：指针张量 gather**——标量展开设计把每个矩阵元素建模为独立 `[BM]` 向量，
  load/store 地址 = `base + k`（stride = n²），机器级落为逐元素标量 LD/ST 序列，
  这是 `_sk6`/`_sk8` SCALAR 占比 32.3%/36.6% 与 MTE2/3 段式搬运的共同根源。
- **越界行隔离记录**：热行全部落在 kernel 源码区间（_sk6:344-405 / _sk8:626-817），
  无跨 kernel/host 侧错位行；`Copy1D.cpp`/`DMAUtils.h`/`VecUtils.h` 为编译器运行库外部行，
  已按口径标注 ⚠外部、不作优化靶点。
- **修复方向**（按成因）：访存连续化（block_ptr / 行 tile 连续 load 替代 stride-gather）。
  **先验实验证据**（`experiments/CLASS_OPTIMIZATION.md`，同 kernel 字节级代码上测得）：
  n=8 行粒度 `_skrow8` 28.70µs vs `_sk8` 31.24µs（**-8.1%** ✅）；n=16 `_skrow16` 52.56µs
  vs scale 57.73µs（**-8.9%** ✅）；n≤6 行粒度反而更慢（n=4：35.3µs vs 17.6µs ❌，
  小 tile layout/mask 开销 > gather 代价）——**连续化收益拐点在 n≥8**。

## 5. 优化建议（按优先级）

| # | 建议 | 判定依据（字段=值 @ 产物路径） | 预期方向 |
|---|---|---|---|
| 1 | **case11 换行粒度 kernel（`_skrow12` 形态）**：`_sinkhorn_scale_masked` 用 N=16 padding 跑 n=12，25% 向量车道恒死 + `axis=1` 列归约树 | vec=94.01%、LDU 冲突=32.00% @ `board/case11_scalem`；官方 80.98µs/0.1235x（全场最差）；先验：v6 实测 `_skrow12` 43.3µs（**-47%**）@ `experiments/CLASS_OPTIMIZATION.md` | VEC 指令总量削减（去 padding + 显式跨行加替代 reduce 树），预估 case11 加速比 0.12→0.23+ |
| 2 | **case12 换 `_skrow16`**：`_sinkhorn_scale` 的 `tl.sum(axis=1)` 列归约是最贵形态（v1 IR 已证列归约 temp_buffer 2×） | vec=97.14%、LDU=40.39% @ `board/case12_scale_n16`；先验：v7 实测 52.56µs（**-8.9%**） | 同上；VEC 已 97% 近终止线，只能换结构 |
| 3 | **case6 治 MTE3 bound + 尾波**：grid=128/56 核 = 2.29 波（核间差异 49.8%），store 为 16 路 stride-gather | mte3=51.51%、LDU=9.88%、aiv_time 24.1~38.0µs @ `board/case6_sk4_bm128` | BM 微调使 grid 对齐整数波（如 grid≈112/168）；store 连续化评估；判定表方案：增大单次搬运量/访存连续化 |
| 4 | **case10 换 `_skrow8`**：SCALAR 36.64% 根源 = IR 证实的 128 个 gather op | IR gather 128 ops @ `ir_output/_sk8_ttir.mlir`；vec=52.04% @ `board/case10_sk8`；先验：-8.1% | 访存连续化（n=8 已越收益拐点） |
| 5 | **case4 类（8 个 case 的基本盘）提核利用率**：grid=32 只用 32/56 核；MTE3 eff 48% 与 RVECSU 41% 相分离 | 掩盖失效 MTE2×MTE3=0.81%、MTE3×RVECSU=1.88% @ `sk4_pipe_overlap.json`；核利用 57% @ `board/case4_sk4_bm32` | 扫 BM=16（grid=64）/BM=18~19 非 2 幂（grid≈56）让核间互相掩盖三段；⚠️ BM 扫描历史只试过 32/64/128（`board_sk4.sh`），BM=16 是未探索点 |
| 6 | case9 保持 `_sk6`：VEC bound 但行粒度先验更差（24.72 vs 19.12µs） | vec=51.76% @ `board/case9_sk6`；先验实验 | 不动作；SCALAR 32.3% 为 gather 固有 |
| 7 | 不建议方向：case11/12 的调度级调优（UB 融合/双流水）——VEC 已 94%/97% 超 80% 终止线；`_sk4` 程序内双流水（单 load→迭代链→单 store 无流水可开） | §3.1/§4.3 | 避免无效立项 |

**理论空间估算**：建议 1+2+4 落地后（先验实测值），case10/11/12 加速比 0.315/0.124/0.269 →
约 0.34/0.23/0.30，20 case 几何平均粗估 +6%~+9%；建议 3/5 为探索性（无先验数值）。

## 6. 循环状态

- 当前轮次：v4sk2 首轮全量分析（r1）；**未触发 10 次规则**（无 bound 路径仅 `_sk4` 一轮仿真即定位三段相分离结构）。
- 终止线核对：case11/12 的 VEC pipe 已 94%/97% ≥80%——这两条路径**达理想终止条件**，
  建议主 agent 接受"只能换 kernel 结构"的结论；其余 kernel 均未接近各自 bound 的理论极限。
- 下一轮建议路径：按 §5 建议 1/2/4 换 kernel（先验证据最强的三项）→ 代码变更后**重新上板**
  判定 bound 转移（旧结论即失效）→ case6/case4 类做 BM 扫描实验（建议 3/5）。
- 归档自检：`python3 check_archive.py <MhcSinkhorn目录>` → **「FAIL: 无迭代目录」**（预期内：
  校验器面向 generator `output/iter_N/` + manifest 指纹链结构，本目录是已交付算子的打榜产物目录，
  无 iter_N 结构可检）。归档未通过项不阻断——本分析为对已交付版本的独立 profiling，
  按 §5.1 精神归档于 `profiling/v4sk2_analysis/`（board ×7 OPPROF + IR ×6 文件 +
  `board_summary.json` / `sk4_pipe_overlap.json` + 驱动 `board_drv.py` / `ir_drv.py` +
  本报告副本），原始采集只增不删。

## 7. 证据附录

**关键 CSV 原始值**（`board/case*/OPPROF_*/PipeUtilization.csv` 中位数）：

| case | aiv_vec_ratio | aiv_scalar_ratio | aiv_mte2_ratio | aiv_mte3_ratio | mte2_active_bw |
|---|---|---|---|---|---|
| 4 | 0.4228 | 0.2554 | 0.2184 | 0.3868 | 26.6 GB/s |
| 6 | 0.4328 | 0.2369 | 0.3504 | 0.5151 | 53.3 GB/s |
| 7 | 0.2939 | 0.1721 | 0.2295 | 0.4177 | 7.5 GB/s |
| 9 | 0.5176 | 0.3230 | 0.1598 | 0.3149 | 51.9 GB/s |
| 10 | 0.5204 | 0.3664 | 0.1384 | 0.3037 | 71.4 GB/s |
| 11 | 0.9401 | 0.1764 | 0.0416 | 0.0200 | 3.7 GB/s |
| 12 | 0.9714 | 0.1296 | 0.0360 | 0.0184 | 9.4 GB/s |

（A5 平台 `aiv_mte3_active_bw` 字段缺失为 NA，属已知平台差异；`Task Duration` 取自 OpBasicInfo.csv。）

**trace 事件片段**（`_sk4` core0.veccore0）：WAIT_FLAG `MTE3←VEC FLAG:0` ×11 / 3.82µs；
`VEC←MTE3` ×4 / 1.12µs；`MTE2←VEC` ×7 / 0.97µs。全量见 `v4sk2_analysis/sk4_pipe_overlap.json`。

**环境附注**：采集用 device0（默认）；`ASCEND_RT_VISIBLE_DEVICES=1` 下 msprof 报
`<StartFFTSTask> Profiling channel start failed` 且解析失败（留档于首轮 board 日志，已删除重采）——
后续采集请勿设该变量。

**数据缺口**：见 §2 末尾四项；`_sk4` BM128 仿真包与 `_sk2` 仿真包如需下钻可经
`profiling/run_sim_prof.sh`（MHC_* 环境变量）复采。
