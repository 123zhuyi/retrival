# 上板数据分析：bound 判定

> 来源：profiling 方案（用户提供）。上板数据直接采集到文件后，即可判断出当前算子
> 是否达成 bound，以及是否存在显著问题。本文件是判定表唯一依据。

数据输入（triton-simulator board 模式产出，`--aic-metrics` 需覆盖三项）：

| CSV | 用途 |
|---|---|
| `PipeUtilization.csv` | pipe 占比与核间耗时 → bound 判定、核间负载不均衡 |
| `ResourceConflictRatio.csv` | 冲突比率 → Bank Conflict / 资源竞争 |
| `L2Cache.csv` | L2 命中率 → 访存局部性问题 |

> 字段名中 `ai*` 表示 `aic`（Cube Core）或 `aiv`（Vector Core），取决于算子类型。
> 字段完整定义（8 类 msprof CSV）查 `ops-profiling/references/`，本文件只列判定用字段。
> **A5 平台（Ascend910_95 / Ascend950，Reg-based）**：本判定表的阈值逻辑沿用，但部分字段
> （活跃带宽、Mix Block Dim）在 A5 上可能缺失或为 N/A，且 CV 融合算子命中 L0C→UB 直通时
> FIXP 占比下降属正常收益——平台差异细则查 `ops-profiling/references/a5-architecture.md`。

## 判定用字段速查

| CSV | 字段 | 含义 |
|---|---|---|
| PipeUtilization | `Task Duration(us)` | Task 总耗时（含调度+执行+响应），与理论耗时对比用 |
| PipeUtilization | `ai*_time(us)` | 各核耗时 → 核间差异 |
| PipeUtilization | `aiv_vec_ratio` | Vector 计算 pipe 时间占比 |
| PipeUtilization | `aic_cube_ratio` | Cube 计算（Matmul）pipe 时间占比 |
| PipeUtilization | `ai*_mte1_ratio` / `ai*_mte2_ratio` / `ai*_mte3_ratio` | 片上/L0 搬入/GM 搬出 pipe 时间占比 |
| PipeUtilization | `ai*_scalar_ratio` | 标量 pipe 时间占比 |
| ResourceConflictRatio | `aiv_vec_total_cflt_ratio` 及各子项 | 见 Bank Conflict 行 |
| L2Cache | `ai*_total_hit_rate(%)` | L2 命中率 |

## bound 判定表

**bound 定义**：某 pipe 的占比是全部 pipe 中**最高**，且 **>50%**（SCALAR 特例
>30%）→ 该 pipe 即当前 bound，优化它性价比最高。**无 bound** = 下表所有 pipe
占比条件均未命中（均 <50%）→ 必须进入仿真数据找大量空泡的成因。

| 问题 | 判定条件 | 子问题 | 子判定条件 | 解决方案 |
|---|---|---|---|---|
| **VEC Bound** | `aiv_vec_ratio` 是最高占比，且 >50% | — | — | 针对 vector 优化：UB 融合、使用 triton-ascend 专属优化接口 |
| **CUBE Bound** | `aic_cube_ratio` 是最高占比，且 >50% | — | — | 针对 cube 优化：使用 L0C 累加减少计算量 |
| **MTE2 Bound** | `ai*_mte2_ratio` 是最高占比，且 >50% | 离散访存 / cache 命中低 | 结合 L2Cache.csv | 针对 MTE2 优化：使用离散访存专属接口、增强 Cache 命中、访存连续化、增大单次搬运量 |
| **MTE3 Bound** | `ai*_mte3_ratio` 是最高占比，且 >50% | — | — | 针对 MTE3 优化：使用离散访存专属接口、访存连续化、增大单次搬运量 |
| **MTE1 Bound** | `ai*_mte1_ratio` 是最高占比，且 >50% | — | — | 针对 MTE1 优化：调整计算顺序减少 L0AB 数据搬运次数 |
| **SCALAR Bound** | `ai*_scalar_ratio` >30% | 标量降级 | 结合 IR 分析 | 针对 scalar 优化：避免标量降级、将输入标量常量化 |
| | | 输入常量未标记 | 代码静态检查 | 同上 |
| **无 Bound** | 上述所有比例均低于 50% | — | — | **必须进入仿真数据**找到大量空泡形成原因：使能双流水、尽量增加并行度 |

## 附带问题（与 bound 判定并行检查，命中即独立立项）

| 问题 | 判定条件 | 子问题 | 子判定条件 | 解决方案 |
|---|---|---|---|---|
| **核间负载不均衡** | PipeUtilization.csv 各核 `ai*_time(us)` 差异 >10% | — | — | 手动启用两个 vector、合理化分核减少尾块空转 |
| **Bank Conflict** | ResourceConflictRatio.csv `aiv_vec_total_cflt_ratio` >5% | bankgroup 冲突 | `vec_bankgroup_cflt_ratio` >3% | 修改访存 stride |
| | | bank 冲突 | `vec_bank_cflt_ratio` >3% | 修改访存 stride |
| | | 资源冲突 | `vec_resc_cflt_ratio` >5% | 调整 Vector/Cube 调度 |
| | | MTE 冲突 | `vec_mte_cflt_ratio` >3% | 调整计算和搬运的调度 |
| **L2 Cache 命中率低** | L2Cache.csv `ai*_total_hit_rate(%)` <50% | — | — | 让访存连续化、一次访存数据量增大 |

## 判定流程

1. 读 `PipeUtilization.csv`，列出全部 pipe 占比（vec / cube / mte1 / mte2 /
   mte3 / scalar），确认 `ai*` 前缀对应的核类型（aic / aiv）。
2. 取**最高占比** pipe：>50%（scalar >30%）→ 命中 bound，记录 pipe 名 + 占比 +
   对应解决方案；均 <50% → 判「无 bound」，转仿真气泡分析。
3. 并行核对附带问题三表（核间差异 / 冲突比率 / L2 命中率），命中项与 bound
   结论一并输出（两者可同时成立，如 MTE2 bound + L2 命中率低互相印证）。
4. SCALAR Bound 必须下钻子问题（IR 分析 / 代码静态检查）后才能给建议——
   标量降级与常量未标记的修复手段不同。
5. 多指标矛盾（如 mte2 与 vec 同为 45%+）时，以最高者为 bound、次高者为候选，
   下一轮 bound 转移时优先复核候选。

## 理论终止条件（80% 理论性能）

理想优化终止条件：某个 pipe 达到 **80% 理论性能**，即

```
算子用时 ≈ 80% × 数据量 / 理论带宽      （搬运类 bound：MTE1/MTE2/MTE3）
算子用时 ≈ 80% × 计算量 / 理论算力      （计算类 bound：VEC/CUBE/SCALAR）
```

用途：给主 agent 的收敛信号——bound pipe 已接近理论极限时，继续优化该 pipe
性价比极低，建议接受现状或换算法结构；未接近时说明仍有空间（对照 bound 类型
的解决方案清单）。
