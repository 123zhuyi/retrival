# Triton 算子性能分析报告：MhcSinkhorn / `_mhc_sinkhorn_fused`

> 按 `ops/triton-profiling-analysis/SKILL.md` §5 结构输出。
> 采集链路：上板 `msprof op`（bound）+ 仿真 `msprof op simulator`（气泡）+ IR（同代码状态）。
> 本报告只做分析，不改代码；优化落地交主 agent / `triton-latency-optimizer`。

## 1. 分析对象与环境

| 项 | 值 |
|---|---|
| 算子 | `level3/mhc_sinkhorn`（MhcSinkhorn，20 轮行列交替归一化） |
| kernel | `_mhc_sinkhorn_fused`（AST 枚举：本实现**唯一**算子 kernel） |
| 实现 | `phase0-7/MhcSinkhorn_generated.py` · sha256 `ab7d2433…`（= 提交版 v1 代码，opt_iter_1） |
| 典型 case | **case 15**（`select_sim_case.py` 工作量中位数）：`[1024,4,4]` f32，`iter=40`，`eps=1e-6`，total_numel=16384 |
| 交叉验证 case | case 4：`[1024,4,4]` f32，`iter=20`，`eps=1e-6`（72µs 平台代表） |
| 器件 | Ascend950PR_9579（DAV_3510），Block Dim = 32 vector core，freq 1650 MHz |
| CANN / Triton | cann-9.1.0 / triton 3.2.0 / torch_npu 2.9.0.post8 |
| 分析目标 | 上板判 bound → 仿真判气泡 → 给优化点 |
| 当前轮次 | r1（首轮 profiling） |

## 2. 数据采集清单

| 数据路 | 采集名 | 产物 | 完整性 |
|---|---|---|---|
| 上板 | `board_case15/OPPROF_2026101912…` | `PipeUtilization.csv` / `ResourceConflictRatio.csv` / `L2Cache.csv` + `OpBasicInfo.csv` | ✅ 32 block 全量 |
| 上板 | `board/OPPROF_2026101912…` | 同上（case 4） | ✅ 32 block 全量 |
| 仿真 | `prof_out__mhc_sinkhorn_fused_case15/OPPROF_20261019123346_CTOFRMCFOCSKJBFI` | 32 core × (`instr_exe.csv` + `code_exe.csv` + `trace.json`) + 总 `trace.json` + `visualize_data.bin` + `dump/` | ✅ complete（64 csv / 33 trace / 2 总产物） |
| 仿真 | `prof_out__mhc_sinkhorn_fused/OPPROF_20261019113901_VWFBEFOPPYWHAJYZ` | 同上（case 4） | ✅ complete |
| dispatch | `sim_meta/sim_dispatch_matrix.json` | 20/20 case 真实探测；全部只发 1 个 kernel | ✅ |
| 选例 | `sim_meta/sim_selection.json` | median → case 15 | ✅ |
| IR | `phase0-7/output/opt_iter_3/ir_output/_mhc_sinkhorn_fused_{ttir,ttadapter,last_pass}.mlir` | 与性能采集**同一代码状态**（sha `ab7d2433…`） | ✅ |

## 3. 上板 bound 判定

### 3.1 各 pipe 占比（32 block 中位数）

| case | Task Duration | `aiv_time` | **vec** | scalar | mte2 | mte3 |
|---|---|---|---|---|---|---|
| **case 15**（iter=40，中位） | **142.49 µs** | 141.39 µs（141.26–141.80） | **99.20%** | 1.83% | 0.28% | 0.15% |
| case 4（iter=20） | 74.13 µs | 72.65 µs（72.04–72.95） | **97.50%** | 2.06% | 0.54% | 0.29% |

### 3.2 附带问题

| 问题 | 判定条件 | 实测 | 结论 |
|---|---|---|---|
| Bank/资源冲突 | `aiv_vec_ldu_cflt_ratio > 5%` | **24.63%**（case15）/ 23.96%（case4） | ✅ **命中**：向量 load 单元冲突显著 |
| 核间负载不均衡 | 各核 `ai*_time` 差异 >10% | 141.26–141.80 µs，**极差 0.38%** | 无 |
| L2 命中率低 | `aiv_read_hit_rate < 50%` | 11.1%（但 MTE2 仅 0.28%，总访存量 ~64 lines/core） | 不立项（绝对量可忽略） |
| 其它 | `aiv_vec_stu_cflt_ratio` 0.07%，`aiv_mte3_wait_ratio` 43%（MTE3 仅 2 条事件） | — | 参考价值低，不立项 |

### 3.3 bound 结论

> **VEC Bound 达成：case15 `aiv_vec_ratio = 99.20%`，case4 = 97.50%**，均为最高占比且远超
> 50% 判定线；**同时超过 80% 理想终止线**。MTE2/MTE3/SCALAR 全部 <3%，不存在搬运或标量 bound。
> 附带 **LDU 冲突 24.63%**（>5%）为独立立项项。

## 4. 气泡分析（仿真，case15 主 / case4 交叉）

> 虽然已达成 VEC bound，仍按要求做了气泡定位，用于确认"是否还有空泡可消"。

### 4.1 每 pipe 空泡（32 核自适应取 core0 两子核，中位数）

| pipe | merged（队列占用） | wait | eff（有效计算） | 空泡率 | 成因 |
|---|---|---|---|---|---|
| **RVECST**（向量执行流） | **97.7%** | 0.0% | **97.7%** | 2.3% | 少量节拍碎片 |
| SCALARLDST | 2.8% | 0 | 2.8% | 97.2% | 物理空闲（期间 RVECST 活跃） |
| SCALAR | 1.0% | 0 | 1.0% | 99.0% | 同上 |
| MTE2 | 0.9% | 0 | 0.9% | 99.1% | 同上 |
| VECTOR（同步队列） | 0.9% | 0.9% | 0.0% | — | 仅 WAIT_FLAG |

### 4.2 掩盖矩阵与 WAIT_FLAG

- `RVECST × SCALARLDST` 重叠 **64.45%**、`RVECST × SCALAR` **24.15%** → 掩盖成立；无 ≥1% 活跃 pipe 对落入 <10% 失效区。
- WAIT_FLAG 全 trace 仅 3 条、合计 **0.66 µs**（MTE2←VEC 2 条、MTE3←VEC 1 条），无等待热点。
- case4 交叉验证：RVECST 91.9%，`MTE2×RVECST` 5.18% 标记失效，但 MTE2 总活跃仅 3.0%（0.38µs/板），不构成瓶颈。

### 4.3 结论

> **无气泡可消**：向量执行流已 97.7% 占用、无 WAIT_FLAG 热点、核间零失衡。
> 剩余 2.3% 为不可约的节拍碎片。后续优化**不属于消气泡路径**。

## 4.5 IR 分析（非 SCALAR bound，作为归约形态佐证）

产物：`opt_iter_3/ir_output/_mhc_sinkhorn_fused_last_pass.mlir`（同代码状态）。

| IR 操作 | 位置 | temp_buffer | 说明 |
|---|---|---|---|
| `hivm.hir.vreduce <max>` `reduce_dims=[2]` | L27 首轮 row_max | 8192×f32 = 32 KB | 最连续轴归约 |
| `hivm.hir.vreduce <sum>` `reduce_dims=[2]` | L40 / L66 | 8192×f32 = 32 KB | 行归约（便宜） |
| `hivm.hir.vreduce <sum>` `reduce_dims=[1]` | L53 / L74 | **16384×f32 = 64 KB（2×）** | **列归约（贵）** |
| 标量降级 | i64 出现 3 处、i32 14 处，无指针张量 gather | — | 未见 i32 比较标量降级（EXACT 分支已消除） |

关键证据：**列归约（`reduce_dims=[1]`）的临时量是行归约的 2 倍（64KB vs 32KB）**，且 tile 物理布局为
`memref<512x4x4xf32, strided<[32,8,1]>>`（末轴 4→8 lane 填充），列方向跨行 → 额外的 shuffle 与 LDU 冲突，
与 §5/§6 的热行完全对应。

## 5. 优化建议（按优先级）

| # | 建议 | 判定依据（字段=值 @ 产物） | 预期方向 |
|---|---|---|---|
| **1** | **列归约改为静态展开向量加**：`N` 为 constexpr（≤16），用 `tl.static_range` 把 `tl.sum(x, axis=1)` 展开成 N 次整块 `+`，避免 `vreduce reduce_dims=[1]` 的 2× temp 与跨行 shuffle | code_exe `L146`=**33.4%**、`L137`=3.6% @ case15；IR `reduce_dims=[1] temp=16384xf32`；`aiv_vec_ldu_cflt_ratio`=**24.63%** | 砍掉列归约慢路径 + 降 LDU 冲突；单项预期 -15%~-25%；**❌ 实测无法编译（Triton 不支持 3D 常量切片）** |
| **2** | **消掉每轮 2 次逐元素除法**：改用累积缩放因子迭代（维护行/列缩放向量，`x` 保持 `x0·R·C`，每轮只做加权归约，末轮一次乘回） | code_exe `L147`=**20.8%** + `L145`=**15.2%** + `L138`=3.4% + `L135`≈2% ⇒ **≈36% cycles 来自 div**；`aiv_vec_ratio`=99.2% 说明 div 就是主成本 | 预期 -25%~-35%；**❌ 实测仅 -1.9%（v2，20/20 通过）**：div 不是主成本，见 §5.1 |
| **3** | 复核 grid/核数利用率（32 program / 56 vector core = 57%） | `OpBasicInfo Block Dim=32`；`BB` 受"2 的幂 + grid≤56"夹死（B=1024 → BB=32/16 两档，16 会到 64>56） | 结构性受限；仅多 kernel 分裂/多 case 特化可解，收益有限，**低优先级** |
| 4 | 不做：消气泡 / 双流水 / 搬运优化 | MTE2 0.28%、WAIT_FLAG 0.66µs、RVECST 97.7% | 无收益 |
| 5 | 不做：标量降级 / 边界冗余（#6/#16） | scalar 1.83%；EXACT 分支已消除 i32 LT | 无收益 |
| 6 | 每轮收尾仍执行 IR 分析（#31） | — | 上述 1、2 落地后重采 IR 复核归约形态 |

> 建议 1、2 为**初版推断**；r2 实验已证伪其中两条（见 §5.1）：建议 1 无法编译，建议 2 实测仅 -1.9%。
> 「-35%~-50%」的理论区间**不成立**。

### 5.1 实验验证（r2，已落地并实测）

实验目录 `../experiments/r2_colsum_unroll/`（报告 `REPORT.md`），基线 case4（iter=20, 74.13µs）：

| 变体 | 改动 | 精度 | Task Duration | 判定 |
|---|---|---|---|---|
| v1 | 建议 1：列归约静态展开 | — | — | ❌ Triton 不支持 3D 常量切片，**无法编译** |
| **v2** | 建议 2：`x/s` → `x*(1/s)` | 20/20 | **72.72 µs** | ✅ 仅 **-1.9%**（vec 97.50%→98.60%） |
| v3 | 列归约走「转置→连续归约→转回」 | 20/20 | 81.40 µs | ❌ +9.8%，回退 |

**关键修正**：`code_exe` 里 L145+L147 的 ≈36% cycles **不是除法吞吐**——把 tile 级除法换成
"小向量倒数 + tile 乘"只省了 1.9%，说明这些 cycles 主要花在**归约结果的布局转换与广播写回**。
行级 cycles 不能直接当作运算类型成本，必须用改动实验验证（本轮已完成）。

→ 三条低风险改写（归约形态 / 除法形态 / 转置归约）**全部试完，无 ≥5% 增量**；
VEC bound 98.6%、内轴仅 N=4（末轴填充到 8 lane）是算法特征，本算式已到向量极限。

## 6. 循环状态

- 当前轮次：**r1**（首次 profiling，无历史基准）。
- **VEC bound 99.2% 已越过 80% 理论终止线** → 按 skill 判定，该实现"向量 pipe 已接近理论极限"，
  消气泡与调流水无意义；**下一次优化只能是"降低每元素向量运算量"的归约/算法重写**（建议 1、2）。
- 10 次规则：未触发（无"无 bound 消气泡"循环）。
- bound 转移预期：v2（72.72µs）与 v1 相比 vec 占比反而升到 98.60%，**bound 未转移**；
  本算式下的微优化已收敛，后续只有算法/精度级重构或接受现状（见 §5.1）。

## 7. 证据附录

**A. 上板原始值（中位数，32 block）**

```
case15: aiv_time=141.39us  aiv_vec_ratio=0.9920  aiv_scalar_ratio=0.0183
        aiv_mte2_ratio=0.0028  aiv_mte3_ratio=0.0015  Task Duration=142.49us
        aiv_vec_ldu_cflt_ratio=0.2463  aiv_vec_stu_cflt_ratio=0.0007  aiv_vec_wait_ratio=0.0723
case4 : aiv_time=72.65us   aiv_vec_ratio=0.9750  aiv_scalar_ratio=0.0206
        aiv_mte2_ratio=0.0054  aiv_mte3_ratio=0.0029  Task Duration=74.13us
        aiv_vec_ldu_cflt_ratio=0.2396
```

**B. 仿真热行（case15, core0.veccore0, 总 637,659 cycles）**

| 行 | 代码 | cycles 占比 | calls |
|---|---|---|---|
| L146 | `col_sum = tl.sum(x, axis=1, keep_dims=True)` | **33.4%** | 30,452 |
| L147 | `x = x / (col_sum + eps)` | **20.8%** | 21,115 |
| L145 | `x = x / (row_sum + pad_row)` | 15.2% | 18,260 |
| L144 | `row_sum = tl.sum(x, axis=2, keep_dims=True)` | 14.7% | 14,287 |
| L137/L138 | 首轮 col_sum / div | 3.6% / 3.4% | — |
| L134/L127 | 首轮 row_sum / row_max | 2.9% / 2.1% | — |

**C. 仿真 pipe 指令分布（case15, core0）**

```
RVECLD 35.5% | RVECEX 24.0% | RVECST 13.6% | PUSHQ 12.7% | RVECSU 11.3% | VECTOR 1.6%
top 指令: RV_SMOV 3675, RV_SADDI 2962, RV_SADD 2151, RV_VLDAS 2040, RV_VLDUI 2040,
          RV_VSTUI 1417, RV_VSEL 1024, RV_VSTAI 971, RV_SCMP 652, RV_VDIV 650
```

**D. 数据缺口（如实标注）**

1. 采集未走 `triton-simulator/scripts/collect.sh`（无 SIGINT 部分采集窗口、无 `stale__` 归档），
   而是用等价的 `run_sim_prof.sh` 一次性跑完（真实 case 在窗口内完成，产物齐全，无截断偏置）；
   `capture_status.json` / `collection_meta.json` / `sim_index.json` 由产物补齐/手写。
2. 仿真采了两套 case（case15 中位主集 + case4 交叉验证），目录名保留 `prof_out__<kernel>` 以符合
   lc5 形式，与 skill 的 `<tag>__<kernel>` 命名不同。
3. IR 复用 `opt_iter_3` 的转储（代码 sha 与采集一致），未在本轮重新提取。
4. 建议 1/2 均为**分析结论**，尚未落地；落地后必须重采上板 + 仿真 + IR 才能判定 bound 转移。
