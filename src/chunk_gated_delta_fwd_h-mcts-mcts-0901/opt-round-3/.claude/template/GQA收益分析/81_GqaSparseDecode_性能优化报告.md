# 81_GqaSparseDecode Triton-Ascend 性能优化最终报告

> 状态：**迭代进行中**，本文随轮次更新。第 6 节（流水采集）按知识库规定放到优化完成后再做。

## 1. 达成值

| 项 | 值 |
|---|---|
| 基线 speedup_vs_torch | **0.2233**（旧工具链 CANN 8.5.1 记为 0.215；本机 9.1.0 复测口径） |
| 当前最好 speedup_vs_torch | **1.3791**（`v2_blockd128`，50/50 全过） |
| 相对基线提升 | **6.18x** |
| 目标 3x | **尚未达成**，见第 7 节 |
| 精度 | **50/50**（`verify.py` 原判据，未改阈值，未用 `--verify_not_required` 取正式结论） |
| 交付版本 | `output/best_generated_code.py`（= `work/v2_blockd128.py`，取自 best 轮而非最后一轮） |
| 测量口径 | `triton-op-verifier/scripts/` 的 `verify.py` + `benchmark.py --keep_res`（warmup 5 / repeats 20），全部 50 case |
| NPU | 卡 1，Ascend910B2C，24 aicore / 48 vectorcore |
| 工具链 | triton 3.2.0 / triton_ascend 3.2.2 / CANN 9.1.0 / bishengir-compile 1.2.0 (8796a8ac1508, 2026-07-29, llvm 19.1.7) |

> ⚠️ **口径说明**：`benchmark.py` 的 `parse_operator_latency` 按 **kernel 名** groupby，
> 同名 kernel 一次调用里发射 N 次只按一次计。本实现全程只有 **1 个 kernel**，
> 所以测得时延就是该 kernel 的设备侧 duration，被如实计量；
> torch 参考侧是几十个不同名小 kernel 的「单次平均」之和，**与 case 规模几乎无关**
> （50 个 case 的 framework 时延全都落在 62~130 us）。
> ⇒ 本算子的 speedup 曲线基本等于「fw 常数 ÷ 我方 kernel 时间」，
> **case 越大越吃亏**，这是本算子难以靠单 kernel 拿到高分的结构性原因。
> 所有正式结论均按此口径给出。

## 2. 基础设施

沿用 `$ROOT/_infra/`（四算子共享）：

1. **`_infra/toolchain.sh`** — 非交互 shell 读不到 `~/.bashrc`（第 6 行 `[ -z "$PS1" ] && return` 守卫），
   会静默继承容器层的旧 CANN 8.5.1 + bishengir 0.1.0。所有 verify/benchmark/IR dump 前必须 source。
2. **`_infra/iter.sh`** — 单轮闭环 verify → benchmark `--keep_res` → 瓶颈画像；精度不过直接终止不测性能。
3. **`_infra/bench_clean.sh`** — 三道闸门：benchmark 全局互斥锁 `/tmp/gqa_bench.lock`
   （80/81 的 torch golden 是三重 python 循环、CPU 密集，并发测量会同步抬高分子分母，
   speedup 这种比值型指标不能自证有效）；必须测满 50 case；工具链指纹前后对照。
4. **`_infra/envstamp.py`** — 每轮校验 triton / CANN / bishengir 版本与 libtriton 指纹，版本不对非零退出。
5. **本算子自建（`work/diag/`）**：
   - `gen.py` —— 复刻 `get_input_groups()` 的**单个 case** 输入构造（不构造其余 49 个），
     诊断脚本据此秒级拿到与 verify 完全一致的输入。
   - `local.py` —— 本地复刻 verify.py 的三项判定（allclose / matched_ratio≥0.9 / MERE<rel_thr），
     用于在正式 verify 之前秒级筛掉精度不过的变体。
   - `sweep.py` —— npu Event 设备侧计时 + constexpr 组合扫描，一次编译多种变体只跑主导 case。
     **诊断性**，正式结论一律回到 `benchmark.py`。
   - `probe_iter_cost.py` / `probe_pv_cost.py` / `probe_dma.py` —— 成本拆解探针（见第 3 节）。
   - `var_bisect.py` —— 精度故障的单变量二分。
   - `sim_prec.py` —— CPU 侧精度方案模拟（不占 NPU，用来先否掉精度方案）。

## 3. 瓶颈画像与根因

### 3.1 画像（v2_blockd128，耗时加权，头部 27 个 case 覆盖 81% 耗时）

```
aic_mac_ratio     = 0.0522     <- Cube 算力几乎空转
aic_scalar_ratio  = 0.5705
aic_mte2_ratio    = 0.1633
aiv_vec_ratio     = 0.1178     <- 向量单元也没饱和
aiv_scalar_ratio  = 0.5329
cube_utilization  = 86.06%
串行度 (aic+aiv)/Duration = 1.738   <- Cube 与 Vector 基本不重叠
50 case 合计 kernel 时间 = 5087 us
```

### 3.2 成本反推：耗时严格按「页」线性，与 case 规模无关

从 v2 逐 case 反推 **耗时 ≈ 每页代价 × ceil(programs/24) × 每 program 页数**：

| case | programs | 每 program 页数 | 实测 us | 折合 us/页 |
|---|---|---|---|---|
| 5 | 2 | 1 | 6.7 | 6.7 |
| 30 | 40 | 4 | 48.3 | 6.0 |
| 42 | 256 | 4 | 255.3 | 6.0 |
| 45 | 96 | 16 | 374.5 | 5.9 |
| 24 | 80 | 16 | 335.9 | 5.2 |

**固定启动开销几乎为 0，全部成本按页线性，每页 5~6 us。**
这与 28_FlashAttentionFwd 的结论（每次 kv 迭代 4~6 us，且几乎与 tile 大小、dtype 无关）
是同一现象。⇒ **全局目标函数 = 最小化「循环迭代总数 × 每迭代握手代价」。**

### 3.3 每页 5~6us 花在哪（`diag/probe_iter_cost.py`，96 program × 16 页合成 kernel）

| 逐段加料 | us/页 |
|---|---|
| 只算地址 | 0.80 |
| + load K | 0.62 |
| + load K,V + QK dot | 1.54 |
| + softmax 向量段 | 1.50 |
| **+ PV dot（fp32 / 单原生 / p_hi+p_lo 双原生）** | **8.85 / 10.56 / 9.84** |
| 把 PV 的 p 换成**循环不变量**（切断 Cube→Vector→Cube 依赖） | 3.06 |
| 只有 QK dot | 2.30 |

`diag/probe_dma.py`（全部喂 Cube、不经 Vector）：

| load 形态 | us/页 | 有效带宽 |
|---|---|---|
| 只读 K（每行取前 256B，行距 512B） | 0.575 | 1367 GB/s |
| **读 K + V（两次跨步半行）—— 现状** | **1.692** | 929 GB/s |
| **一次连续读整页 `[BS, 2D]`** | **0.642** | 2452 GB/s |

**根因（两条互相印证）**：

1. **主导项是「Cube 出 scores → Vector 出 p → Cube 算 PV」这条每页一次的跨核往返**：
   把 PV 的 `p` 换成循环不变量后从 8.85 掉到 3.06 us/页，而两个操作数都来自 GM/循环不变量的
   双 dot 只要 1.69 us/页。**代价不在 dot 本身，在于 dot 消费了向量段产出的操作数。**
2. **不是访存受限也不是算力受限**：实测有效带宽仅 ~260 GB/s（v2 全局），
   算术强度 16 flop/byte ≪ 910B 机器平衡点 ~235 flop/byte，
   `aic_mac_ratio` 0.05 / `aiv_vec_ratio` 0.12 两条流水线都远未饱和。
   `aic_scalar_ratio` 0.57 与 `aiv_scalar_ratio` 0.53 是**同步等待被计入标量管线**
   （与 MQA、28 号算子的结论一致），不是真的标量运算过多。

### 3.4 访存下界（先算账）

每个 (token, kv_head) 要读 rtopk 个 128 行页的 K 与 V，即整页 `128×2D×itemsize`；
50 个 case 合计 18398 页 ≈ 1.18 GB，**且没有冗余**（每个 (t,kh) 读的页互不重复）。
按 fw/3 为 3x 所需时延反推：

| 假设 | 预测 geomean 上界 |
|---|---|
| 0.8 TB/s + 20us 固定开销 | 3.39 |
| 1.2 TB/s + 15us | 4.83 |
| 1.6 TB/s + 10us | 6.79 |

⇒ **3x 在访存下界之内可达**，方向是「贴近访存下界」而不是「减少访存」。
其中 case 24/37/41/42/43/44/45 这 7 个即使打满带宽也够不到 3x（例如 case 45 要读 100.7 MB，
1.6 TB/s 下也要 63 us，而 fw/3 只有 41 us），所以 3x 必须靠其余 43 个 case 的几何平均撑起来。

## 4. 有效改动（按贡献排序）

| 改动 | 增益 | 依据什么数据 |
|---|---|---|
| **任务单元 (token, q_head) → (token, kv_head)**，q tile 变 `[BLOCK_G, D]`，QK/PV 都走 `tl.dot` | 0.2233 → 1.3791（**6.18x**） | 基线画像 `aiv_scalar 0.672 / aic_mac 0.036 / 串行度 1.97`；基线 `kh = h // g_size` 让同一 kv_head 组内 G 个 q head 各自把同一份 `[128,2D]` KV 页重读一遍，实测 G∈{4,6,8,16}，主导耗时的 case 37~45 全是 G=16 ⇒ KV 访存被放大 16 倍；且基线 PV 段用 `tl.sum(p[:,None]*v, axis=0)` 广播归约（全在 Vector 上），QK 段 `tl.dot` 的 M==1 |
| **`BLOCK_D` 下限 128**（精度修复，非性能项） | 44/50 → **50/50** | 见第 5 节第 1 行 |

（后续轮次结论见 `81_GqaSparseDecode_性能优化迭代日志.md`，本表随之更新。）

## 5. 已排除方向及其证据

| 方向 | 结论 | 证据 |
|---|---|---|
| **「K/V 同时无掩码 load 触发 bishengir 崩溃」**（80 号算子的历史结论，本算子的首选假设） | **证伪** | 孤立复刻 V 的加载形态（KV 行 `[2D]`、V 取后 D 列、行 stride=2D）共 16 种组合（D∈{64,128} × maskV × fp32dot × 是否伴随 QK dot）全部 `NaN=0, maxerr<1.6e-5`（`diag/probe_pv.py`）；在真 kernel 上换成 80 号已验证的「K 掩码+where / V 无掩码」写法（`var_bisect.py` V3）**仍然 NaN**，且 K/V 都加掩码（V4）NaN 反而更乱 |
| **D=64 整组 NaN 的真根因** | **定位**：`tl.dot` 的「**fp32 操作数 + N=64 输出宽度 + 运行期 trip-count 的 `scf.for` 内**」三者同时满足时被 bishengir 1.2.0 误编译 | 现象刻画：每个 (t,h) 恰好 32/64 个 NaN，**全部落在偶数列，奇数列是 0~0.8 的垃圾正数** ⇒ 是 `[·,64]` 输出 tile 的交织排布被写坏。单变量二分：BLOCK_G 16→4 仍 NaN（与 padding 无关）；PV 改原生 dtype dot → NaN=0；BLOCK_D 64→128 → NaN=0；循环 trip 改常量 1 → NaN=0。BRIEF §4.2「fp32 tl.dot 可用且逐位一致」的探针是**循环外**测的，适用条件应收窄为「循环外 或 N≥128」 |
| `m_new = -inf ⇒ exp(-inf-(-inf))=NaN` / `l_i==0 ⇒ acc/l_i` / `rtopk==0` / BLOCK_G padding 污染 dot 归约 | **排除** | 这些若成立会与 D 无关地在 D=128 上同样发作，而 44 个 D=128 的 case 全过；且 V5/V2 两个只动 dot 形态的变体直接全绿 |
| **单次低精度 PV dot**（去掉 `v.to(fp32)`，`dot(p.to(v.dtype), v)`） | **精度证伪** | 49/50 过，**case 42 MERE = 9.004e-3 > bf16 阈值 7.812e-3**。p 舍入到 bf16 的半 ulp 2^-9 在 `Σp·v / Σp` 上被放大后越阈 |
| **「分子分母口径一致」代替双 dot**（`l_i` 也用同一份被舍入的 p̃ 求和，指望权重扰动一阶项相消） | **精度证伪** | CPU 全量模拟 50 个 case（`diag/sim_prec.py`）：case 42 MERE 1.30e-2，比不一致口径的 1.31e-2 几乎没改善。而 p_hi/p_lo 双 dot 是 1.23e-5 |
| **`p_hi + p_lo` 双原生 dtype dot 替代 fp32 dot** | **性能证伪**：0.7089 vs v2 的 1.3791 | 精度极好（case42 MERE 1.03e-5），但循环内 dot 从 2 个变 3 个，实测时间正好翻倍（5087→10500 us），`aiv_scalar_ratio` 0.533→0.904。**反过来验证了「循环内 dot 的个数才是成本单位」** |
| **NP 页用「行基址向量」gather 合并成一个大 tile**（把每页一次的 CV 往返摊薄到 1/NP） | **性能证伪**：0.9995 | 逐 case 看 NP 确实有效（NP=1 的 case 比值 0.52、NP=2 约 0.5~0.87、NP=4 达 0.74~0.95），但 `aiv_mte2_ratio` 从 0.081 暴涨到 **0.539** —— **gather 形态的 load 走不了 Cube 的 GM→L1 直载，被迫由 AIV 搬进 UB 再转交**，省下的握手被多出来的搬运吃光 |
| **split-K（flash-decoding）+ merge kernel** | 先算账，暂缓 | benchmark 按 kernel 名 groupby，多一个**不同名** kernel 会给**每个 case** 都加一次它的单次平均时延；而低并行度只出现在 programs<24 的约 20 个 case 上，其余 30 个 case 只会净亏 |

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
| 旧（按 kernel 名，只计一次） | 0.1044 ms | 0.0757 ms | 1.3791 | 50/50 |
| **新（按单次调用，全部计入）** | **3.1449 ms** | **0.0750 ms** | **41.9099** | **50/50** |

**impl 几乎不变（0.0757 → 0.0750），变的全是 torch 那一侧**（0.104 → 3.14 ms，约 30 倍）。
参考实现是 `for pid_b in range(total_q)` + `for kh in range(KH)` 的双层 Python 循环。
⇒ 旧口径下推出的「本算子天花板只有 1.95x」必须重算：那套成本模型
（每页 5.81us、dot 个数是成本单位、UB 188/192KB）**约束的是 impl 绝对时间，与口径无关，仍成立**；
失效的只是把它换算成 speedup 的那一步。

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
