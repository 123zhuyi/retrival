# MhcSinkhorn Simulator 流水采集（按 `7_SparseFlashAttention_lc5` 形式组织）

本目录是 MhcSinkhorn 的**指令级流水 / profiling 产物**，目录结构与文件组织逐项对齐
`/data/s30073567/cannbench/profiling/7_SparseFlashAttention_lc5`：

- 每个 kernel 一个 `prof_out__<kernel>/`
- 其下 `OPPROF_<ts>_<id>/` 分 **`dump/`** 与 **`simulator/`** 两部分
- `simulator/` 下是**每个 core 一个目录**（`coreN.veccoreM/`，含 `trace.json` + `instr_exe.csv` + `code_exe.csv`），
  外加**总 `trace.json`** 与**总 `visualize_data.bin`**
- `dump/` 保留 lc5 的三件套：`aicore_binary.o` / `object_dump.txt` / `pc_start_addr.txt`

## 目录

```
profiling/
├── prof_mhc_sinkhorn.py              采集驱动（单次 forward；B/n/iter/eps 可用 MHC_* 覆盖）
├── run_sim_prof.sh                   msprof op simulator 采集脚本（位置参数式调用）
├── analyze_sim.py                    单包结构校验 + pipe 占比 / 热源码行汇总
├── README.md                         本文件
├── analysis_report.md                ★ 上板 bound + 仿真气泡 + 优化点（按 triton-profiling-analysis §5）
├── diagnostic_report_sim.md          单包采集报告（自动生成，结构校验 ✅）
├── bubble_analysis.json             case4：pipe 掩盖度 + WAIT_FLAG 归因原始结果
├── bubble_analysis_case15.json      case15：同上
├── sim_meta/                         triton-simulator 数据包元数据
│   ├── sim_dispatch_matrix.json     20/20 case 真实 dispatch 探测（全部只发 1 个 kernel）
│   ├── sim_selection.json           典型 case 选择：median → case 15
│   ├── sim_case___mhc_sinkhorn_fused.py  选例生成的真实 case 驱动
│   └── sim_index.json               workload ↔ 产物总清单（手写汇总，见 §数据缺口）
├── board/                           case4 上板 msprof op（PipeUtilization/ResourceConflictRatio/L2Cache）
├── board_case15/                    case15（median）上板 msprof op
├── logs/                            采集环境 / msprof 日志 / 气泡分析日志 / 失败尝试留档
├── prof_out__mhc_sinkhorn_fused/            （case4 iter=20 仿真包；磁盘 100% 已满，已删除释放 695MB，
│                                                结构同 case15，可由 run_sim_prof.sh 复采）
├── prof_out__mhc_sinkhorn_fused_case15/     ★ median case15（iter=40）仿真包，结构同上
└── ../experiments/r2_colsum_unroll/         优化实验：v2 倒数-乘（-1.9%，20/20）、v3 转置归约（回退）
    ├── REPORT.md                            实验报告（含 v1 不可编译的证伪）
    ├── variants/                            v1/v2/v3 变体代码
    ├── verify/                              精度验证（v2/v3 均 20/20）
    └── board_v2/ · board_v3/                变体上板 profiling
```

> 说明：本 kernel 是纯 vector（RegBase DAV_3510）实现，因此 `simulator/` 下只有
> `veccore*`、**没有** `cubecore*`；这与 lc5 中 `_sfa_gather_kv_kernel` 只有 veccore 的情形一致。

## 与 lc5 的对照

| 项 | lc5（SparseFlashAttention） | 本目录（MhcSinkhorn） |
|---|---|---|
| 顶层组织 | `prof_out__<kernel>/OPPROF_*/` | 同 |
| 每 kernel 子目录 | `dump/` + `simulator/` | 同 |
| dump 内容 | `aicore_binary.o` / `object_dump.txt` / `pc_start_addr.txt` | 同（msprof 额外产生的空 `*.toml` 已按 lc5 口径剔除） |
| simulator 每 core | `core*.*/` + `trace.json` + `instr_exe.csv` + `code_exe.csv` | 同 |
| simulator 根 | `trace.json` + `visualize_data.bin` | 同 |
| kernel 数 | 4（gather_kv / gathered_attn / split_attn / split_combine） | **1**（`_mhc_sinkhorn_fused`，本算子唯一 kernel） |
| 采集开关 | `msprof op simulator --launch-count=5` | `--launch-count=1`（单次 launch，产物即 lc5 的单实例形态） |
| 采集包 | 4 个 kernel 各 1 包 | 2 包：case15（median, iter=40，主）/ case4（iter=20，交叉验证） |

## 采集命令与环境

```bash
# 容器：zmm-agent（远端 NPU 宿主 90.90.93.29）
cd output1014/MhcSinkhorn/profiling
bash run_sim_prof.sh                                     # case4（默认 iter=20）
MHC_ITER=40 OUT=$PWD/prof_out__mhc_sinkhorn_fused_case15 bash run_sim_prof.sh   # median case15
# 等价展开：
msprof op simulator \
    --output=prof_out__mhc_sinkhorn_fused \
    --kernel-name=_mhc_sinkhorn_fused \
    --launch-count=1 \
    --soc-version=Ascend950PR_9579 \
    --timeout=30 \
    /root/miniconda3/envs/py311/bin/python3 prof_mhc_sinkhorn.py
```

环境与版本见 `logs/collect_env.txt`（CANN 9.1.0 / triton 3.2.0 / torch_npu 2.9.0.post8 /
Ascend950PR_9579 / 仿真库 `simulator/dav_3510`）。

> **踩坑记录**：本版 msprof 的 `--application=` 只接受 ELF 可执行文件，传 shell 脚本会报
> `CheckInputFileValid ... is not a file` + `Failed to load libruntime_camodel.so`；
> 正确做法是用**位置参数** `<python> <script.py>`（脚本里已按此实现）。

## 结论（详见 `analysis_report.md`）

### 上板 bound（`msprof op`，32 block 中位数）

| case | Task Duration | vec | scalar | mte2 | mte3 | LDU 冲突 |
|---|---|---|---|---|---|---|
| **case15（median, iter=40）** | **142.49 µs** | **99.20%** | 1.83% | 0.28% | 0.15% | **24.63%** |
| case4（iter=20） | 74.13 µs | 97.50% | 2.06% | 0.54% | 0.29% | 23.96% |

→ **VEC Bound，且已越过 80% 理想终止线**；核间极差 0.38%（零失衡）；LDU 冲突 >5% 独立立项。

### 仿真（case15）

- RVECST（向量执行流）队列占用/有效 **97.7%**，MTE2 0.9%、SCALAR 1.0%，WAIT_FLAG 全 trace 仅 3 条 0.66µs
- → **无气泡可消**；剩余 2.3% 为节拍碎片
- 热行：`L146` 列归约 **33.4%**、`L147` 除法 **20.8%**、`L145` 除法 15.2%、`L144` 行归约 14.7%
- IR 佐证：`vreduce reduce_dims=[1]`（列归约）temp_buffer 64KB = 行归约（32KB）的 **2 倍**

### 优化点（分析结论，未落地）

| # | 建议 | 依据 | 实验结果 |
|---|---|---|---|
| 1 | 列归约改 `static_range` 静态展开 | L146 33.4% + LDU 24.6% + IR 2× temp | ❌ **无法编译**（Triton 3D 常量切片不支持） |
| 2 | 消掉逐元素除法（`x/s` → `x*(1/s)`） | L145+L147 ≈ 36% | ✅ 20/20 通过，**仅 -1.9%**（74.13→72.72µs） |
| 2b | 列归约走「转置→连续归约→转回」 | 同上 | ❌ +9.8%（81.40µs），回退 |
| 3 | grid 32/56 核利用率（结构性受限） | Block Dim=32，BB 被"2 的幂 + ≤56"夹死 | 未测（结构性受限） |
| — | 不做：消气泡 / 双流水 / 搬运 / 标量降级 | MTE2 0.28%、scalar 1.83% | 无收益 |

> v2 的 -1.9% 说明 `code_exe` 的 36% 不是除法吞吐，而是归约结果的布局/广播；行级 cycles 必须靠改动实验验证。
> 详见 `../experiments/r2_colsum_unroll/REPORT.md`。

## 校验与复现

```bash
python3 analyze_sim.py .        # 逐项校验 dump/simulator/每-core/总 trace+bin，缺项退出码 1
```

当前结果：`[structure] OK`（dump 3/3、simulator 总 trace + bin、32 个 core 目录、64 个 csv）。

> 体积说明：case4 包约 695 MB，case15（iter=40）包约 2.4 GB（trace/bin 随迭代数近似翻倍）。
> `dump/` 已按 lc5 口径只保留三件套；如需更小体积，可用 `MHC_B=256` 或 `MHC_ITER=20` 复采，
> 或删掉交叉验证包 `prof_out__mhc_sinkhorn_fused/`（结论以 case15 为准）。
>
> 上板数据（`board*/`）由 `msprof op --aic-metrics=PipeUtilization,ResourceConflictRatio,L2Cache`
> 采集，命令见 `logs/board_prof.log` / `logs/board_case15.log` 开头；分析结论见 `analysis_report.md`。
