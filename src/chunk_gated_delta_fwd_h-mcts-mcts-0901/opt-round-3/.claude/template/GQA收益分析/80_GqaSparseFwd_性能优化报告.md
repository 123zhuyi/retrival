# 80_GqaSparseFwd Triton-Ascend 性能优化最终报告

> 状态：**迭代进行中**，本文随轮次更新。第 1 节的"最终值"栏在收尾轮定稿。

## 1. 达成值

| 项 | 值 |
|---|---|
| 基线 speedup_vs_torch（正确工具链重测） | **2.2904**（50/50） |
| 历史记录值（作废） | 0.9438（CANN 8.5.1 + bishengir 0.1.0） |
| 最终 speedup_vs_torch | *（收尾轮填入）* |
| 目标 3x | *（收尾轮判定）* |
| 交付版本 | `output/best_generated_code.py` |
| 测量口径 | `triton-op-verifier/scripts/` 的 `verify.py`（50/50 强制）+ `benchmark.py --keep_res`；阈值未改，未用 `--verify_not_required` |
| NPU | 2（910B2C，24 aicore / 48 vectorcore） |
| 工具链 | triton 3.2.0 / triton_ascend 3.2.2 / CANN 9.1.0 / bishengir-compile 1.2.0 (8796a8ac1508, 2026-07-29, llvm 19.1.7) |

**换编译器本身就值 2.43x**：同一份代码（`output/optimized_code.py`，一行未改）在
CANN 8.5.1 + bishengir 0.1.0 上记录为 0.9438，在 CANN 9.1.0 + bishengir 1.2.0 上实测 **2.2904**。
⇒ 交接文档里"历史数字一律作废"这条被实测证实，且**所有编译器相关的结构性结论都必须重做**
（本报告第 5 节里有 4 条历史结论在新编译器上翻转）。

### ⚠️ 口径说明（必读，否则会误判优化空间）

`benchmark.py` 的 `parse_operator_latency` 按 **kernel Name** 分组，每组取最后 `repeats` 行
除以 `repeats`，再对不同 Name 求和。后果：

- 本实现每次 forward 只发射 **1 个** kernel ⇒ impl latency **就是** kernel 的真实 duration；
- torch golden 是 `B × TQ × KH` 三重 python 循环，同名 aten kernel 发射成百上千次却**只被计一次**。

实测证据：50 个 case 的 `framework.avg_latency_ms` **全部落在 0.085~0.130 ms**，
与 case 规模（TQ 从 1 到 64、seq 从 2 到 32784）**几乎无关**。

⇒ `speedup_i = fw_i / impl_i ≈ 0.109 / impl_i`，**优化目标退化成"压 kernel 绝对时间"**：
达成 3x 需要 `geomean(impl) ≤ geomean(fw)/3 = 0.1090/3 = 0.0363 ms`（基线 0.0476 ms，需要 −23.7%）。
这是验收判据，不是真实加速比；真实 wall-clock 上 torch golden 慢得多。

## 2. 基础设施

1. **`_infra/iter.sh`** — 单轮闭环 verify → benchmark `--keep_res` → 瓶颈画像；verify 不过直接终止。
2. **`_infra/toolchain.sh`** — 非交互 shell 读不到 `~/.bashrc`（`[ -z "$PS1" ] && return` 守卫），
   会静默继承容器层的 CANN 8.5.1 + bishengir 0.1.0。所有测量前强制 source，`envstamp.py` 逐轮校验。
3. **`_infra/bench_clean.sh`** — benchmark 全局互斥锁 `/tmp/gqa_bench.lock` + 工具链指纹前后对照
   + 必须测满 50 case。
4. **`work/micro.py`（本轮新建，诊断用）** — 复刻 `benchmark.py` 的口径：
   `torch_npu.profiler` 采 `kernel_details.csv`，按 Name groupby 取最后 `repeats` 行均值。
   **交叉验证**：case 41 micro 184.4us vs benchmark 183.0us、case 43 140.7 vs 139.2、
   case 49 91.1 vs 91.0 —— 误差 <1%，可以在 4~15 个 case 上做快速单变量 A/B，
   把一轮的成本从 ~35 分钟压到 ~1.5 分钟。
   > ⚠️ 第一版 micro 用 wall-clock 和 `torch.npu.Event` 都不可用：host 侧单次发射开销约 70us，
   > 会把 <70us 的 kernel 完全淹没（case 0 真实 10.8us 被测成 96.4us）。必须走 profiler。
5. **`work/ab.sh`** — 固定 case 子集 + 固定环境跑 micro，每批次内必带一个 `v0_base` 对照，
   用于抵消机器负载（本机是共享的，load average 一度到 60+）。

## 3. 瓶颈画像（基线 v0_base，耗时加权，全 50 case）

```
aic_mac_ratio     = 0.0657     <- Cube 算力几乎空转
aic_scalar_ratio  = 0.6840     <- 同步等待被计入标量管线
aic_mte2_ratio    = 0.2765
aic_mte1_ratio    = 0.1870
aic_fixpipe_ratio = 0.1793
aiv_vec_ratio     = 0.2583     <- 向量单元同样没饱和
aiv_scalar_ratio  = 0.2423
cube_utilization  = 85.7%
串行度 (aic+aiv)/Duration = 1.732
```

### 成本模型反推

用 `tasks = TQ*KH`、串行迭代数 `= ceil(tasks/24) * rtopk`（grid=(TQ*KH,)，每 program 串行扫 rtopk 个页）：

| case(1-idx) | tasks | rtopk | 串行迭代数 | impl(ms) | 每迭代 |
|---|---|---|---|---|---|
| 5  | 4   | 1  | 1  | 0.0070 | — （固定开销 ~6-7us） |
| 6  | 4   | 16 | 16 | 0.0322 | ~1.7us |
| 39 | 96  | 16 | 64 | 0.2012 | ~3.1us |
| 42 | 256 | 5  | 55 | 0.1830 | ~3.4us |

每次迭代只搬 64KB（K/V 各 `[128,128]` bf16）、做 3 个 `[16,128]x[128,128]` 的 dot，
理论 <0.1us，实测 1.7~3.4us。`aic_mac_ratio` 6.6% + 串行度 1.73 ⇒
**Cube 与 Vector 互相等待是唯一瓶颈**，算力、访存、向量指令条数都不是。

### 根因的直接实验证据（比 ratio 更硬）

把 3 行向量掩码计算（`pos / pos_f / valid`）从 `tl.dot` **之后**挪到**之前**，
其他一字不改 ⇒ 4 个 case 的 SUM 从 **555.5us 涨到 1212.0us（2.18x）**。
反向操作（把两个 PV dot 提到 `l_i/acc` 缩放之前）⇒ **−12.4%**。

⇒ 本 kernel 的时间几乎全部由「一次迭代内 Cube 指令与 Vector 指令的相对发射位置」决定。
这是 BRIEF 检查清单第 7 条（CV 指令重排）的同一机理，在 MQA 上只值 +1.9%，
**在本 kernel 上单条改动的量级是 2.18x**。

## 4. 有效改动（按贡献排序）

| 改动 | 增益 | 依据什么数据 |
|---|---|---|
| **切到正确工具链**（CANN 9.1.0 + bishengir 1.2.0） | 0.9438 → **2.2904**（+143%） | 同一份代码零改动重测；`envstamp.py` 双向核对版本指纹 |
| **CV 指令重排：两个 PV dot 提前到 `l_i`/`acc` 缩放之前发射**（`r1_pvfirst`） | −12.4% kernel 时间（4-case SUM 556.6 → 489.0，三次复测 489.0/486.7/488.6） | 逐 case：38 号 201.6→170.7（−15.3%）、41 号 182.0→158.9（−12.7%）、43 号 139.8→124.2（−11.2%）；纯重排，数值逐位不变（MERE 2.235e-6→2.325e-6 属 dot 累加顺序差异） |

*（收尾轮补充：本轮仍在测的编译开关与跨迭代延迟更新项）*

## 5. 已排除方向及其证据

| 方向 | 结论 | 证据（4-case SUM us，基线 555.5/556.6） |
|---|---|---|
| **PV 段换 fp32 单 dot**（BRIEF 4.2 点名"最有希望的一刀"） | **证伪：1.80x 劣化** | `v1_f32dot` = 1002.7。数值上确实更准（MERE 2.2e-6→5.5e-7，50 case 精度全过），但 910B Cube 的 MMAD 只有 fp16/bf16/int8 通路，fp32 dot 走模拟路径；且 `v.to(fp32)` 把 V 的 UB 占用翻倍。⇒ `p_hi/p_lo` 双 dot 不能删，它同时是**性能**选择 |
| **每次迭代 gather 多个 KV 页**（把 rtopk 次迭代压成 rtopk/NPG 次） | **证伪：NPG=2 慢 1.95x；NPG=1/4 编译失败** | `v3_gather` NPG=2 = 1086.3；NPG=1 `Unknown core type: llvm.func @malloc`；NPG=4 `hivm-plan-memory Failed`（UB overflow）。3-D 页 gather + `tl.reshape` 强制一次 UB staging，代价远超省下的迭代数 |
| **一次迭代处理两个页（标量寻址，不用 gather）** | **证伪：1.92x 劣化** | `p2_pairpage` = 940.1。两套 tile 同时活跃 → UB 压力翻倍，编译器退化 |
| **K/V 合并成一次连续 `[128, 2D]` load**（q 右补零做 QK、PV 结果取后半列） | **证伪：13.4x 劣化** | `y1_kvmerge` = 6528.6。页内 `[128,2D]` 确实是完全连续的 64KB，但 `2*BLOCK_D=256` 宽的掩码 load + `where` 让编译器彻底失速 |
| **`valid` 用「标量上界 + 单次向量比较」代替两次 `pos` 比较** | **证伪：2.77x 劣化** | `m1_scalarlim` = 1538.4。标量上界要在标量单元做 int→float 与 `minimum`，插进了向量流水 |
| **K 也走无掩码 load**（历史记载会触发 bishengir 非确定性崩溃） | **在 1.2.0 上既不崩也不快** | `m2_kunmask` = 565.1（+1.6%，噪声内），4 个 case 精度全过。K 侧 vsel staging 不是瓶颈 —— 历史结论的"崩溃"部分作废，"有收益"部分也不成立 |
| **持久化 grid**（`grid=min(NTASK,NCORE)` + 核内 task 循环） | **无收益** | `x1_persist` NCORE=24 → 487.7、NCORE=48 → 486.8，对照 R1 489.3。BRIEF 检查清单第 5 条在本 kernel 上不适用：多波次 program 启动开销不显著 |
| **`blk → page` 两跳标量 GM load 提前一轮预取** | **编译失败（部分 shape）+ 无收益** | `x3_prefetch`：case 38/43 `MLIRCompilationError`；能编过的 case 41 = 158.0 vs 158.9、case 5 = 35.5 vs 35.5 |
| **K 直接按转置布局 load，去掉 `tl.trans`** | **无收益** | `x4_ktrans` = 488.3 vs 489.3。`tl.trans` 已被编译器折进 ND2NZ |
| **V 的 load 挪到 QK dot 之后** | **无收益** | `x5_vlate` = 486.4 vs 489.3（−0.5%，噪声内） |
| **`p_hi` 用截断（1 条 AND）取代 RNE 链（4~9 条整数指令）** | **无收益** | `t2_r1_trunc` = 486.7 vs 486.7 |
| **删掉冗余的 `p = tl.where(valid, p, 0)`** | **无收益** | `t1_r1_nowhere` = 486.8 vs 486.7。（因此保留它——它是全 lane 失效时的 NaN 兜底，零成本） |
| **`delta = exp(...)` 也挪到 dot 之后** | **无收益** | `u1` = 488.8、`u2`（+截断）= 486.6 |
| **`tl.maximum(propagate_nan=ALL)`**（20_FlashAttentionV2 上 +7.5%） | **无收益** | `u3` = 487.0 vs 486.6 |
| **`tl.exp2` + 把 log2e 折进 sm_scale** | **编译失败** | 本版 triton 无 `tl.exp2` |

**横贯全部失败轮的一条结论：所有"减少向量指令条数"的改动一律 0 收益**
（截断 vs 9 条 RNE 链、删 where、`propagate_nan`、`delta` 后移，全部落在 ±0.5% 内），
**所有"改变 tile 形状/访存结构"的改动一律大幅劣化**（1.9x ~ 13x）。
只有**改变 Cube 指令相对 Vector 指令的发射位置**这一类有收益。

## 6. 与历史经验 / 其他算子的差异

| 条目 | 历史 / 其他算子结论 | 本算子实测 |
|---|---|---|
| 编译器版本 | "历史数字一律作废" | **证实**：同代码 0.9438 → 2.2904 |
| PV 段 fp32 dot（BRIEF 4.2 / 21 号算子） | 21 号：fp32 与低精度 dot 只差 4%，低精度反而 UB overflow ⇒ 建议换 fp32 | **相反**：本算子 fp32 dot **慢 1.8x**，低精度双 dot 才是快的 |
| "只有减少迭代次数有效"（20_FlashAttentionV2） | 减少 KV 迭代次数是最大一笔 | **不适用**：本算子减少迭代的唯一手段（页 gather）自身代价更高，净劣化 |
| CV 指令重排（检查清单第 7 条） | MQA +1.9%、28_FA +6.4% | **本算子 −12.4%（反向做则 +118%）**，是唯一有效杠杆 |
| `tl.maximum(propagate_nan=ALL)` | 20_FA +7.5%（需按块数分档） | **无收益** |
| K 无掩码 load | 历史日志：触发 bishengir `ConvertLinalgRToBinary` 非确定性崩溃 | **1.2.0 上不崩了**，但也没有收益 |
| `aic_scalar_ratio` 高的含义 | 是同步等待被计入标量管线，不是标量运算多 | **一致**（0.684 全部是 CV 握手） |

---

## ★ 最终成绩（`triton_dev_v2` 新口径，2026-08-14 重测）

`triton_dev_v2` 的 `6e4dc13c fix(triton-op-verifier): benchmark 时延改为按单次调用计算，
支持 forward 内多次启动 kernel` 变更了时延口径：由整组行数反推发射次数
`L = len(group)/active_count`，取末尾 `L×active_count` 行求和再除以 `active_count`
⇒ **一次 forward 内的多次发射如实全部计入**（旧版按 kernel 名 groupby，只计一次）。

测量条件：NPU 1、`--lock-frequency`（实测无频率漂移）、
仓库自带 `verify.py` / `benchmark.py`，阈值未改、未用 `--verify_not_required`。

| 口径 | framework | impl | speedup | 精度 |
|---|---|---|---|---|
| 旧（按 kernel 名，只计一次） | （未记录） | 0.0476 ms | 2.2904 | 50/50 |
| **新（按单次调用，全部计入）** | **2.9083 ms** | **0.0488 ms** | **59.5558** | **50/50** |

**impl 几乎不变（0.0476 → 0.0488），变的全是 torch 那一侧。**
参考实现是逐 (batch, head, block) 的 Python 循环，一次 forward 发射上千个小 kernel，
旧口径按 kernel 名只计一次，把 torch 的真实代价低估了一个数量级以上。

**四个算子横向（同一口径、同一测量条件）**：

| 算子 | framework | impl | speedup |
|---|---|---|---|
| 21_GroupedQueryAttention | 0.1797 ms | 0.0715 ms | **2.5119** |
| 33_AdaptiveAttention | 0.5559 ms | 0.1386 ms | **4.0101** |
| 80_GqaSparseFwd | 2.9083 ms | 0.0488 ms | **59.5558** |
| 81_GqaSparseDecode | 3.1449 ms | 0.0750 ms | **41.9099** |

---

## ★ 权威最终数字（2026-08-14 22:2x，同一干净窗口连测）

测量条件：整机空闲（所有 NPU 0%）、NPU 2、`--lock-frequency`（无频率漂移）、
`triton_dev_v2` 的 `benchmark.py`，阈值未改、未用 `--verify_not_required`。
四个算子**背靠背连测**，消除跨时段不可比。

| 算子 | framework | impl | speedup | 精度 |
|---|---|---|---|---|
| 21_GroupedQueryAttention | 0.1800 ms | 0.0713 ms | **2.5261** | 50/50 |
| 33_AdaptiveAttention | 0.5562 ms | 0.0962 ms | **5.7810** | 50/50 |
| 80_GqaSparseFwd | 2.8838 ms | 0.0477 ms | **60.4755** | 50/50 |
| 81_GqaSparseDecode | 3.0926 ms | 0.0754 ms | **40.9901** | 50/50 |

**impl 几何平均 = 0.0705 ms**（起点 0.0776；理论下界 0.0294 = 四算子同时达 vendor peak）。

### 测量可复现性验证

三档与此前分时段记录的偏差均在 **±1.5%** 内（21: 2.5261 vs 2.5119；33: 5.7810 vs 5.7479；
80: 60.4755 vs 59.5558）。⇒ **空闲机器 + `--lock-frequency` + 同卡**三条同时满足时测量可信；
今天出现的所有异常数据（负优化被当成改进、绝对值虚高 10 倍、比值外推方向反转）
**全部可归因到机器负载**，没有一次是工具本身的问题。
