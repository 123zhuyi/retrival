# 81_GqaSparseDecode 迭代日志

工具链固定：CANN 9.1.0 / bishengir-compile 1.2.0 (8796a8ac1508) / triton 3.2.0 /
triton_ascend 3.2.2 / 910B2C (num_aicore=24, num_vectorcore=48)。
所有 verify / benchmark 经 `_infra/iter.sh <OP> <impl> <TAG> 1`（NPU 1）。

时延口径提醒（BRIEF §3）：benchmark 按 **kernel 名** groupby 取每次发射平均耗时，
总时延 = Σ(不同名 kernel 的单次平均)。本实现只有 1 个 kernel，所以测得时延 ≈ 该
kernel 的设备侧 duration；torch 侧 framework ≈ Σ(几十个不同名小 kernel 的单次平均)
≈ 0.08~0.125 ms，与 case 规模基本无关。

---

## 访存下界预算（动手前先算账）

每个 (token, kv_head) 任务需要读 rtopk 个 128 行页的 K 与 V，即整页
`128 × 2D × itemsize`。逐 case 统计（`work/diag/` 里的估算脚本）：

| 项 | 值 |
|---|---|
| 50 个 case 合计迭代数 (Σ TQ·KH·rtopk) | 18398 |
| 最大 case (45) 访存 | 100.7 MB |
| 次大 (24 / 42) | 83.9 / 67.1 MB |

以 fw/3 为 3x 所需时延，纯 HBM 带宽下界：

| 假设 | 预测 geomean |
|---|---|
| 0.8 TB/s + 20us 固定开销 | 3.39 |
| 1.2 TB/s + 15us | 4.83 |
| 1.6 TB/s + 10us | 6.79 |

结论：**3x 在访存下界内可达**，且 KV 访存已无冗余（每个 (t,kh) 读的页互不重复），
所以方向是「贴近访存下界」而不是「减少访存」。算术强度 16 flop/byte，机器平衡点
约 235 flop/byte ⇒ **深度访存受限，Cube 空闲**，用 Cube 换 Vector 一律划算。

---

## 轮次记录

### v1_kvgroup_tc（上一位 agent 交接）—— 精度 44/50，拒绝

任务单元 (token, q_head) → (token, kv_head)，q tile `[BLOCK_G, D]`，QK/PV 都走 `tl.dot`。
失败 case：1/2/3/13/15 报 NaN 位置不匹配，5 报 AccuracyError。
共同特征：**全部 6 个 D=64 的 case，无一例外**（D=64 的 case 恰好就是这 6 个）。

### 诊断轮 A —— 定位 D=64 NaN 根因（`work/diag/`）

**现象刻画**（`diag/cols.py`）：每个 (t,h) 恰好 32/64 个 NaN，且 **NaN 落在全部偶数列，
奇数列是 0~0.8 的垃圾正数**。跨 head、跨 case 完全一致 ⇒ 不是掩码/边界问题，
是 **[·,64] 输出 tile 的交织排布被写坏**。

**证伪 1（BRIEF §4.3 首选假设）**：孤立复刻 V 的加载形态（KV 行 `[2D]`、V 取后 D 列、
行 stride=2D），16 种组合（D∈{64,128} × maskV × fp32dot × 是否伴随 QK dot）
全部 `NaN=0, maxerr<1.6e-5`（`diag/probe_pv.py`）。
⇒ **"K/V 同时无掩码 load" 不是本算子的根因**，80 号的历史结论在 81 上不成立。

**单变量二分**（`diag/var_bisect.py`，全部在 case 1 上，只改一处）：

| 变体 | 改动 | 结果 |
|---|---|---|
| V0 | 基线 BG=16 / BD=64 / K,V 无掩码 / fp32 PV dot | NaN=256/512，偶数列 |
| V1 | BLOCK_G 16→next_pow2(G)=4 | 仍 NaN（**证伪**：与 BLOCK_G padding 无关） |
| V2 | PV dot 改原生 dtype（不 `.to(fp32)`） | **NaN=0**，maxabs 3.9e-3 |
| V3 | K 掩码+where / V 无掩码（80 号已验证写法） | 仍 NaN（**证伪 BRIEF §4.3 假设**） |
| V4 | K/V 都掩码+where | 仍 NaN，且 NaN 列变成 0,1,2,3,4,6…（更乱） |
| V5 | BLOCK_D 64→128（D=64 补零） | **NaN=0**，maxabs 1.2e-4 |
| V6 | 循环 trip 改成常量 1 | NaN=0 |

**根因判定**：`tl.dot` 的 **fp32 操作数 + N(=BLOCK_D)=64 输出宽度 + 运行期 trip-count
的 `scf.for` 内**，这三者同时满足时被 bishengir 1.2.0 误编译。
把 N 抬到 128（V5）或把 dot 降回原生 dtype（V2）或消掉运行期循环（V6）都能规避。
BRIEF §4.2「fp32 tl.dot 可用且逐位一致」的探针是**循环外**测的，本轮把适用条件收窄为
**"循环外 或 N≥128"**。

**同时排除**（因为 V5/V2 直接全绿，无需再查）：
`m_new=-inf ⇒ exp(nan)`、`l_i==0 ⇒ acc/l_i`、`rtopk==0`、BLOCK_G padding 污染 dot 归约 ——
这些若成立会与 D 无关地在 D=128 上同样发作，而 44 个 D=128 case 全过。

### v2_blockd128 —— 精度 50/50 ✅

改动（单变量）：`BLOCK_D = max(128, next_power_of_2(D))`。
依据：诊断轮 A 的 V5。预期影响 `gqa_sparse_decode_kernel`（唯一 kernel）。
代价：D=64 的 6 个 case 的两个 dot 各多算一倍（补零），但这 6 个 case 都是最小的
（合计迭代数 118 / 18398 = 0.6%）。

精度：**50/50 全过**。speedup **1.3791**（旧基线 0.2233 → 6.2x），
50 case 合计 kernel 时间 5087 us（旧 55479 us）。
画像：`aiv_scalar_ratio 0.533` / `aic_scalar_ratio 0.571` / `serial_ratio 1.738`。

### 诊断轮 B —— 「每页 ~5.2us」到底花在哪（`diag/probe_iter_cost.py` / `probe_pv_cost.py`）

从 v2 逐 case 反推：**耗时 ≈ 每页代价 × ceil(programs/24) × 每 program 页数**，
每页代价中位数 5.2us，与 case 大小无关（case 5 只有 1 页 ⇒ 6.7us；case 42 共 1024 页、
256 program ⇒ 255us，折合 6.0us/页）。固定启动开销几乎为 0，**全部成本按页线性**。

96 program × 16 页的合成 kernel 上逐段加料：

| 内容 | us/页 |
|---|---|
| 只算地址 | 0.80 |
| + load K | 0.62 |
| + load K,V + QK dot | 1.54 |
| + softmax 向量段 | 1.50 |
| **+ PV dot（fp32 / 单原生 / 双原生）** | **8.85 / 10.56 / 9.84** |
| PV 的 p 换成循环不变量（切断 Cube→Vector→Cube 依赖） | 3.06 |
| 只有 QK dot | 2.30 |

⇒ **每个循环内的 `tl.dot` 自带一份固定的 CV 同步代价（~2.6us），与 tile 大小几乎无关。**
「Cube 出 scores → Vector 出 p → Cube 算 PV」这条每页一次的往返是主导项，
不是 DMA 也不是算力：实测有效带宽只有 ~260 GB/s，算术强度 16 flop/byte ≪ 机器平衡点 ~235。

**推论（后续所有轮次的指导原则）：循环内 dot 的个数才是成本单位，不是 tile 的大小。**

### v3_pvnative —— 精度 FAIL（case 42），拒绝

改动：PV dot 由 `tl.dot(p_f32, v.to(f32))` 改成单次原生 dtype `tl.dot(p.to(v.dtype), v)`。
依据：想去掉 `v.to(fp32)` 这个 [128,128]=16384 元素的向量转换。
结果：**case 42 MERE = 9.004e-3 > bf16 阈值 7.812e-3**，其余 49 个 case 过。
⇒ **证伪「单次低精度 PV dot 精度够用」**：p 舍入到 bf16 的半 ulp 2^-9 在
`Σp·v / Σp` 上放大后越阈。（这条与 80 号算子需要 p_hi/p_lo 拆分的结论一致。）

### v4_pvsplit —— 精度 50/50，性能 **回退到 0.7089**，拒绝

改动（相对 v2 单变量）：PV 由 fp32 dot 改成 `dot(p_hi,v) + dot(p_lo,v)` 双原生 dtype dot。
依据：诊断轮 B 里「PV 的 p 换成循环不变量后 3.06us/页」，判断 CV 往返是主因，
以为双原生 dot 能让 V 全程走 Cube 通路不进 UB。
精度：50/50，MERE 从 v3 的 9.0e-3 掉到 **1.03e-5**（case 42），
⇒ **p_hi/p_lo 拆分在 bishengir 1.2.0 上没有被编译器折叠**（BRIEF 记的 80 号那个
`.to(dtype).to(f32)` round-trip 被消除的坑，本轮用最朴素写法没有复现）。
性能：**0.7089**（v2 是 1.3791），合计 10500us（v2 5087us），`aiv_scalar_ratio`
从 0.533 涨到 **0.904**。
⇒ **证伪「双原生 dot 比 fp32 dot 便宜」**：循环内 dot 从 2 个变 3 个，
按每 dot ~2.6us 的固定 CV 代价算正好翻倍。这一轮反过来**验证**了诊断轮 B 的推论。

### 诊断轮 C —— DMA 形态（`diag/probe_dma.py`，全部喂 Cube 不经 Vector）

| load 形态 | us/页 | 有效带宽 |
|---|---|---|
| 只读 K（每行取前 256B，行距 512B） | 0.575 | 1367 GB/s |
| **读 K + V（两次跨步半行）—— v2~v5 的写法** | **1.692** | 929 GB/s |
| **一次连续读整页 `[BS, 2D]`** | **0.642** | 2452 GB/s |

同样的数据量，拆成两条跨步 DMA 比一条连续 DMA 贵 **2.6 倍**：每行只取 256B 而行距
512B，两条 DMA 各自把整条 line 拉进来。合并可省 ~1.05 us/页。


---

## v7_pvskew —— PV dot 错开一轮打断 CV 依赖 —— 精度 50/50，性能 **1.2647**，拒绝

改动（相对 v2 单变量）：把 PV dot 推迟一轮，使循环内两个 dot 的操作数都来自
"上一轮或更早"，与本轮向量段无数据依赖。dot 个数不变（仍 2 个）。
错位后的在线 softmax 记账先在 CPU 上用 200 组随机用例对拍过
（`scratchpad/skew_algebra.py`：v7 == v2 == 一次性 softmax，max err < 1e-12），
所以精度 50/50 是预期内的。

依据：`diag/probe_iter_cost.py` 里"把 PV 的 p 换成循环不变量"一档从 8.85 降到 3.06 us/页。

结果：**1.3791 → 1.2647**，且 `serial_ratio` 1.738 → **1.707 几乎没动**。

⇒ **证伪「CV 握手由数据依赖驱动」**：握手是**按 dot 结构性插入**的，与操作数来自哪一轮无关。
probe 里那 2.9x 来自别的因素（很可能是循环不变量被外提后 UB 压力下降），不是依赖被打断。
副作用：多出 `v_prev[128,128]` fp32（64KB）循环携带量，把本已 188/192KB 的 UB 挤爆。

## v8_pvf16 —— PV dot 由 fp32 换 fp16 原生 —— 精度 50/50，性能 **1.3088**，持平

改动（相对 v2 单变量）：`dot(p_f32, v.to(f32))` → `dot(p.to(f16), v.to(f16))`。

依据：v3 的单次原生 dot（p 降到 bf16）只超阈 **1.15 倍**（MERE 9.004e-3 vs 7.812e-3），
而 fp16 尾数 11 bit 比 bf16 的 8 bit 多 3 bit；V 侧 bf16→fp16 在本算子取值域内
**逐位无损**（300 组随机样本实测偏差 0.0）。

结果：精度 **50/50**（假设成立），性能 1.3088，与 v2 持平 —— 符合"dot 个数不变则时间不变"。
真正的价值是 **UB 从 188KB 降到 156KB，腾出 32KB**，这是 v9 整页方案的前置条件。

## v9_fullpage_f16 —— 整页连续 load —— 精度 50/50，性能 **更慢**，拒绝

改动（相对 v8 单变量）：K/V 两次跨步半行 load 合并成一次整页连续 load
（q 补零成 `[BLOCK_G, 2D]` 让 `dot(q2, kv^T) == q·K^T`；`dot(p, kv)` 的后 D 列即 `p·V`）。

依据：`diag/probe_dma.py` 实测两次跨步半行 1.692 us/页 vs 一次连续整页 0.642 us/页（差 2.6 倍）。

**第一次测量（NPU 0）得 1.6007，是污染数据**：当时机器被另一会话占满
（`msopprof simulator` + 另一个 verify，NPU 0 AICore 93%），framework 从 0.105ms
飙到 0.367ms，impl 从 0.076ms 飙到 0.230ms，绝对值整体放大 3 倍。

**干净重测（NPU 1 + `--lock-frequency`，新版 benchmark）**：

| 版本 | impl | speedup |
|---|---|---|
| **v2_blockd128** | **0.0750 ms** | **41.9099** |
| v9_fullpage_f16 | 0.0956 ms | 31.0008 |

⇒ **证伪整页 load**：省下的 DMA 抵不过两个 dot 的 K 维/N 维翻倍带来的开销
（`acc` 也从 `[16,128]` 变成 `[16,256]`）。**81 的 best 仍是 v2_blockd128。**

⇒ 教训：**共享机器上任何没有环境闸门的测量都不可信**。这一轮差点让一个负优化被固化成 best。

## ★ 口径变更：新版 benchmark 下 81 = 41.91x

`triton_dev_v2` 的 `6e4dc13c fix(triton-op-verifier): benchmark 时延改为按单次调用计算，
支持 forward 内多次启动 kernel` 把时延口径改了：由整组行数反推发射次数
`L = len(group)/active_count`，取末尾 `L×active_count` 行求和再除以 `active_count`
⇒ **一次 forward 内的多次发射如实全部计入**。

对本算子的影响是**决定性**的：torch 参考实现是 `for pid_b in range(total_q)` +
`for kh in range(KH)` 的双层 Python 循环，一次 forward 发射上千个小 kernel。
旧口径按 kernel 名只计一次，把 torch 的真实代价低估了约 30 倍：

| 口径 | framework | impl | speedup |
|---|---|---|---|
| 旧（按 kernel 名，只计一次） | 0.1044 ms | 0.0757 ms | 1.3791 |
| **新（按单次调用，全部计入）** | **3.1449 ms** | 0.0750 ms | **41.9099** |

impl 侧几乎不变（本算子只有 1 个 kernel、只发射 1 次），**变的全是 torch 那一侧**。
⇒ 之前"81 的天花板只有 1.95x"那套推算是在**旧口径**下做的；换成新口径后目标早已达成。
但那套成本模型（每页 5.81us、dot 个数是成本单位、UB 188/192KB）**本身仍然成立**，
它约束的是 impl 的绝对时间，与口径无关。
