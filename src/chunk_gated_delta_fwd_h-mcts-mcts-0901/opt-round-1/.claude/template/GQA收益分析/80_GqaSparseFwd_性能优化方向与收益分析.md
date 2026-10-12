# 块稀疏 GQA prefill 类算子 Triton-Ascend 优化方向与收益分析

> 基于 80_GqaSparseFwd 实测（910B2C，CANN 9.1.0 + bishengir-compile 1.2.0，triton-ascend 3.2.2）。
> 与 20_FlashAttentionV2 / 28_FlashAttentionFwd / 22_MQA 的经验有多处**直接冲突**，冲突处以本文实测为准，
> 并在第 4 节逐条对照。

---

## 0. 一页速查

| 判据 | 本算子的答案 |
|---|---|
| 瓶颈类型 | **Cube↔Vector 跨核串行**（`aic_mac_ratio` 0.066、串行度 1.73、`aic_scalar_ratio` 0.684 全是握手等待） |
| 唯一有效杠杆 | **循环体内 Cube 指令相对 Vector 指令的发射位置** |
| 单条改动的量级 | 反向做（掩码提到 dot 前）**+118%**；正向做（PV dot 提到缩放前）**−12.4%** |
| 减少向量指令条数 | **0 收益**（5 次独立实验，全部 ±0.5% 内） |
| 改 tile 形状 / 访存结构 | **一律劣化 1.9x ~ 13x**（6 次独立实验） |
| 编译开关 | **0 收益**（5 个逐个单开，全部 ±0.5% 内） |
| 减少循环迭代次数 | **不适用**：唯一手段（多页 gather）自身代价更高 |

---

## 1. 算子形态识别（决定哪些经验能套）

块稀疏 GQA prefill 的形态与稠密 FlashAttention 有三点本质差异，直接决定了经验的可迁移性：

1. **KV 不连续**。每个 (token, kv_head) 的 KV 页由 `topk_idx → block_table` 两跳间接寻址得到，
   页与页之间在 GM 里完全无关。⇒ FA 里"放大 BLOCK_KV 让每次迭代多扫一片连续 KV"这条
   最大收益项**在这里没有对应物**：想合并两页只能做 gather，而 gather 自身的代价更高（见 3.2）。
2. **M 维极小且被结构锁死**。任务单元是 (token, kv_head)，dot 的 M = G = H/KH ∈ {4,6,8,16}，
   连 Cube 的一个 16x16 fractal 都填不满。⇒ `aic_mac_ratio` 只有 6.6%，
   **任何以"提高算力利用率"为目标的改动都不会有收益**。
3. **每次迭代的工作量固定**（一页 = 128 个 KV 位置，`BLOCK_N = BS = 128` 被页大小锁死）。
   ⇒ tile 三个维度 (BLOCK_G / BLOCK_D / BLOCK_N) 全部被输入结构锁死，没有 tile 扫描空间。

**识别方法**：先算「每次迭代的理论时间」与「实测时间」的比值。本算子每次迭代搬 64KB、
做 3 个 `[16,128]x[128,128]` 的 dot，理论 <0.1us，实测 1.7~3.4us ⇒ **20~30 倍差距，
一定是握手/同步，不是算力也不是带宽**。这个反推比任何 ratio 都硬。

---

## 2. 有增益的方向

### 2.1 ★★★ 切到正确的工具链（前置条件，不算优化）

同一份代码在 CANN 8.5.1 + bishengir 0.1.0 上 speedup 0.9438，
在 CANN 9.1.0 + bishengir 1.2.0 上 **2.2904**（+143%）。

⚠️ 非交互 shell 读不到 `~/.bashrc`（第 6 行 `[ -z "$PS1" ] && return` 守卫），
会**静默**继承容器层的旧 CANN。两套都能跑通、都不报错。必须每轮打指纹校验。

### 2.2 ★★★ CV 指令重排：把 Cube 的 dot 尽量早发射（本算子唯一有效项）

```python
# ❌ 原式：两个 PV dot 之前先做 l_i / acc 的向量缩放
l_i = l_i * delta + tl.sum(p, axis=1)
acc = acc * delta[:, None]
acc += tl.dot(p_hi, v_tile)
acc += tl.dot(p_lo, v_tile)

# ✅ 改式：先把两个 dot 发出去，再做与之无关的向量缩放
pv = tl.dot(p_hi, v_tile) + tl.dot(p_lo, v_tile)
l_i = l_i * delta + tl.sum(p, axis=1)
acc = acc * delta[:, None] + pv
```

- 纯重排，数值等价（`l_i`/`acc` 的更新本来就不依赖 `pv`）。
- 实测 **−12.4%**（4-case SUM 556.6 → 489.0，三次复测 489.0 / 486.7 / 488.6）。
- 逐 case：38 号 201.6→170.7（−15.3%）、41 号 182.0→158.9（−12.7%）、43 号 139.8→124.2（−11.2%）。

**反向对照（这条比正向更有说服力）**：把 3 行掩码计算
（`pos = blk*BS + n_offs` / `pos_f = pos.to(f32)` / `valid = 两次比较`）
从 `tl.dot` 之后挪到之前，其余一字不改 ⇒ **555.5us → 1212.0us（2.18x 劣化）**。

⇒ **判别方法**：在 kernel 里画一条"Cube 指令 / Vector 指令"的时间轴，
任何夹在两个 Cube 指令之间、且与后一个 Cube 指令**无数据依赖**的向量语句，
都应该往后挪到 Cube 指令之后。这条在 MQA 上只值 +1.9%、在 28_FA 上 +6.4%，
**在块稀疏 GQA 上值 2.18x**，因为这里每次迭代的 Cube 工作量太小，
一次多余的 CV 握手就能把整个迭代的时间翻倍。

---

## 3. 无增益 / 负收益 / 编译失败的方向（这一节比第 2 节更值钱）

所有数字都是 4 个主导 case（0-idx 38/41/43/5）的 profiler kernel duration 之和，
基线 v0_base = 555.5 / 556.6us（两次复现），每批次内都带同一个对照组以抵消机器负载。

### 3.1 ✗✗ PV 段换 fp32 单 dot —— **1.80x 劣化**（与 BRIEF 建议相反）

`v1_f32dot` = 1002.7us。BRIEF 4.2 实测「`tl.dot` 支持 fp32 操作数且与 torch fp32 matmul
`max_abs_err == 0`」，据此建议删掉 `p_hi/p_lo` bitcast 拆分改用一次 fp32 dot。

**数值部分属实**（MERE 从 2.235e-6 降到 5.545e-7，50 case 精度全过），**性能部分相反**：

- 910B 的 Cube MMAD 只有 fp16/bf16/int8 通路，fp32 dot 走模拟路径；
- `v_tile.to(tl.float32)` 把 V 的 UB 占用翻倍（D=128 时 64KB → 128KB），
  与任何"每次迭代多放一份数据"的改动叠加即 `hivm-plan-memory Failed`。

⇒ **`p_hi/p_lo` 双 dot 不是精度妥协，它同时是性能上的正确选择。**

### 3.2 ✗✗ 每次迭代 gather 多个 KV 页 —— **1.95x 劣化 / 编译失败**

动机是"减少迭代次数"（20_FlashAttentionV2 上最有效的一条）。三种规模：

| NPG | 结果 |
|---|---|
| 1 | 编译失败 `[ERROR] Unknown core type: llvm.func @malloc(i64)` |
| 2 | 1086.3us（1.95x 劣化），50 case 精度全过 |
| 4 | 编译失败 `IR Dump After PlanMemory Failed (hivm-plan-memory)` |

写法是 `pages[:,None,None]*stride + n[None,:,None]*stride + d[None,None,:]` 的 3-D gather load
再 `tl.reshape` 成 `[NPG*128, D]`。3-D gather 会强制一次 UB staging，代价远超省下的迭代数。

**用标量寻址做同一件事（一次迭代读两页，两次独立 `tl.load`）同样劣化 1.92x**
（`p2_pairpage` = 940.1us）—— 两套 tile 同时活跃，UB 压力翻倍。

⇒ 「减少迭代次数」这条经验在块稀疏算子上**不可迁移**，因为减少迭代的唯一手段自身更贵。

### 3.3 ✗✗ K/V 合并成一次连续 load —— **13.4x 劣化**

KV cache 布局 `[nb, kh, 128, 2*head_dim]`，页内 `[128, 2D]` 是**完全连续的 64KB**，
而当前 K/V 分两次读是 256B 片段 + 512B 跨步。写法：q 右侧补零到 2D 做 QK
（`dot(q_pad[G,2D], kv^T[2D,128]) == q@K^T`），PV 用 `dot(p, kv[128,2D])` 取后半列。

实测 `y1_kvmerge` = 6528.6us。`2*BLOCK_D = 256` 宽的掩码 load + `tl.where` 让编译器彻底失速。
⇒ **访存连续性不是本算子的瓶颈**，为它付出的 tile 变宽代价是灾难性的。

### 3.4 ✗✗ `valid` 用「标量上界 + 单次向量比较」代替两次比较 —— **2.77x 劣化**

```python
# ❌ lim 是标量, 需要在标量单元做 int->float 与 minimum
lim_f = tl.minimum(seq_f, q_abs_f + 1.0) - (blk * BS).to(tl.float32)
valid = n_offs_f < lim_f
```
`m1_scalarlim` = 1538.4us。标量浮点链插进向量流水，代价远超省下的一条向量比较。
**在本编译器上，标量单元做浮点运算要极力避免。**

### 3.5 ✗ 减少向量指令条数 —— 五次独立实验全部 0 收益

| 改动 | 指令数变化 | 结果 |
|---|---|---|
| `p_hi` 用截断（1 条 AND）代替 RNE 链 | −4（bf16）/ −9（fp16） | 486.7 vs 486.7 |
| 删掉冗余的 `p = tl.where(valid, p, 0)` | −1 个 `[G,N]` vsel | 486.8 vs 486.7 |
| `delta = exp(...)` 挪到 dot 之后 | 0（纯位置） | 488.8 vs 486.7 |
| `tl.maximum(propagate_nan=ALL)` | −6（IR 层展开） | 487.0 vs 486.6 |
| `tl.exp2` + log2e 折进 scale | −1 个 `[G,N]` 乘 | **编译失败**（本版 triton 无 `tl.exp2`） |

`aiv_vec_ratio` 只有 0.258，向量单元根本没饱和。
⇒ **与 20_FlashAttentionV2 的结论一致：向量指令条数不是判据。**
（附带结论：既然截断和 RNE 一样快，就该保留更准的 RNE；既然删 `where` 不快，
就该保留它作为全 lane 失效时的 NaN 兜底。）

### 3.6 ✗ 编译开关 —— 5 个逐个单开，全部 0 收益

| 开关 | 结果 |
|---|---|
| `enable_hivm_auto_cv_balance=True`（名字直指本算子瓶颈） | 487.8 vs 486.7 |
| `ops_reorder=True` | 485.3 vs 486.7 |
| `enable_ubuf_saving=True` | *（见迭代日志）* |
| `enable_preload=True` | *（见迭代日志）* |
| `sync_solver=True` | *（见迭代日志）* |

与 BRIEF「约 30 个试下来命中率很低」一致。

### 3.7 ✗ 持久化 grid（`grid=min(NTASK, NCORE)` + 核内 task 循环）

`x1_persist`：NCORE=24 → 487.7、NCORE=48 → 486.8，对照 R1 489.3。
BRIEF 检查清单第 5 条在本算子上**不适用**：最大 case 的 grid 是 256（10.7 波），
但多波次的 program 启动开销并不显著。

### 3.8 ✗ 跨迭代软件流水（把向量更新延迟一轮 / 预取下一轮的 blk-page）

- `z1_deferred`（`l_i`/`acc` 的更新延迟一轮，与下一轮的 QK dot 重叠）：**4 个 case 全部编译失败**。
- `x3_prefetch`（`blk → page` 两跳标量 GM load 提前一轮）：case 38/43 `MLIRCompilationError`，
  能编过的 case 41 = 158.0 vs 158.9、case 5 = 35.5 vs 35.5（无收益）。

⇒ 与 BRIEF 的提醒一致：**有效的是单次迭代内部重排，迭代之间做流水不行**。
本算子上不仅没收益，还会触发编译器错误。

### 3.9 ✗ K 无掩码 load —— 历史"崩溃"结论作废，但也没有收益

`m2_kunmask` = 565.1（+1.6%，噪声内），4 个 case 精度全过。
历史日志记载 K 无掩码会触发 bishengir `ConvertLinalgRToBinary` 非确定性崩溃
（0.1.0 上"K 无掩码 + G=8 + D=128 抽签式崩溃"）——**在 1.2.0 上不崩了**，
但 K 侧的 vsel staging 本来就不是瓶颈，所以也没有收益。

### 3.10 ✗ 去掉 `tl.trans(k_tile)` / V 的 load 后置

- `x4_ktrans`（K 直接按 `[D, N]` 转置布局 load）= 488.3 vs 489.3 —— `tl.trans` 已被编译器折进 ND2NZ。
- `x5_vlate`（V 的 load 挪到 QK dot 之后）= 486.4 vs 489.3（−0.5%，噪声内）。

---

## 4. 与其他算子经验的冲突对照

| 条目 | 其他算子结论 | 80_GqaSparseFwd 实测 | 冲突原因 |
|---|---|---|---|
| PV 段 fp32 dot | 21 号：fp32 与低精度只差 4%，低精度反而 UB overflow | **fp32 慢 1.8x** | 21 号的 dot 规模大得多，模拟路径的固定开销被摊薄 |
| 减少迭代次数 | 20_FA：最大一笔收益 | **不可迁移** | 稠密 FA 可以直接放大 BLOCK_KV；块稀疏只能 gather，gather 自身更贵 |
| CV 指令重排 | MQA +1.9%、28_FA +6.4% | **−12.4%（反向 +118%）** | 本算子每次迭代的 Cube 工作量极小（mac 6.6%），一次多余握手就能翻倍 |
| `propagate_nan=ALL` | 20_FA +7.5%（需分档） | **0 收益** | 本算子 `tl.maximum` 只作用在 `[16]` 上，展开成 7 条也无所谓 |
| K 转置态 load | 28_FA +0.7% | **0 收益** | — |
| grid 收缩到核数 | 28_FA +0.5% | **0 收益** | — |
| 历史结论会作废 | — | **证实**：同代码 0.9438 → 2.2904，且 4 条历史结构性结论翻转 | 换了 5 个月 + 一个 llvm 大版本的编译器 |

---

## ★ 测量口径变更与最终成绩（`triton_dev_v2`，2026-08-14 重测）

`triton_dev_v2` 的 `6e4dc13c fix(triton-op-verifier): benchmark 时延改为按单次调用计算，
支持 forward 内多次启动 kernel` 变更了时延口径：由整组行数反推发射次数
`L = len(group)/active_count`，取末尾 `L×active_count` 行求和再除以 `active_count`
⇒ **一次 forward 内的多次发射如实全部计入**（旧版按 kernel 名 groupby，只计一次）。

测量条件：NPU 1、`--lock-frequency`（实测无频率漂移）、仓库自带 `verify.py` / `benchmark.py`，
阈值未改、未用 `--verify_not_required`。

| 本算子 | framework | impl | speedup | 精度 |
|---|---|---|---|---|
| 旧口径 | （未记录） | 0.0476 ms | 2.2904 | 50/50 |
| **新口径** | **2.9083 ms** | **0.0488 ms** | **59.5558** | **50/50** |

**impl 几乎不变（0.0476 → 0.0488），变的全是 torch 那一侧。**
参考实现是逐 (batch, head, block) 的 Python 循环，一次 forward 发射上千个小 kernel，
旧口径按 kernel 名只计一次，把 torch 的真实代价低估了一个数量级以上。
⇒ 本文件里「换对工具链 0.9438 → 2.2904」那条结论**不受影响**（同口径对比）；
受影响的只是把 2.2904 当成「离 3x 还差多少」的那类推算。

### 四个算子横向（同一口径、同一测量条件）

| 算子 | framework | impl | speedup |
|---|---|---|---|
| 21_GroupedQueryAttention | 0.1797 ms | 0.0715 ms | **2.5119** |
| 33_AdaptiveAttention | 0.5559 ms | 0.1386 ms | **4.0101** |
| 80_GqaSparseFwd | 2.9083 ms | 0.0488 ms | **59.5558** |
| 81_GqaSparseDecode | 3.1449 ms | 0.0750 ms | **41.9099** |

### 这条对「以后怎么优化」的两点影响

1. **旧口径下的「合并同名 kernel 刷分」「拆成不同名 kernel 会被罚」两条顾虑全部作废**，
   新口径下按真实 wall-clock 设计即可；
2. **旧口径下算出的天花板/缺口必须重算**。成本模型本身（每次 KV 迭代的固定 CV 握手、
   UB 预算、tile 分档）约束的是 **impl 的绝对时间，与口径无关，仍然成立**；
   失效的只是把它换算成 speedup 的那一步。
   ⇒ 本文件前面各节的 speedup 一律是**旧口径**，只用于**方案之间的横向比较**（impl 绝对时间没变），
   **不要与上表的绝对分数混用**。

### 经验已沉淀为可复用 template

本算子族的完整经验已写成 `cannbot-skills_5171` (`triton_dev_v2` 分支) 的
`plugins-official/triton-op-generator/template/gqa_sparse.md`，并接入 AGENTS.md 的
category 判定表（排在 `flash_attention` 之前，从上到下匹配）。
已用 Kimi 驱动 `triton-op-generator` 做端到端验证。

---

## ★★ 时延目标下的补充（2026-08-14，目标改为 impl 几何平均 ≤ 0.03 ms + speedup > 2x）

### 先算账：0.03 ms 等价于「四个算子同时达到 vendor peak」

| 算子 | impl 几何均值 | 理论下界 | 离下界 |
|---|---|---|---|
| 21_GroupedQueryAttention | 0.0715 ms | 0.0259 ms（vendor peak TFLOPS + 3 次 launch） | **1.99x**（逐 case 几何平均） |
| 33_AdaptiveAttention | 0.1386 → **0.0969** | 0.0307 ms | 3.16x |
| 80_GqaSparseFwd | 0.0488 | — | — |
| 81_GqaSparseDecode | 0.0750 | 0.0191 ms（1.18GB KV @1.2TB/s） | 3.9x |

**四者全部打到下界的几何平均 = 0.0294 ms。** 单是 kernel 启动开销的几何平均下界就有 0.005 ms
（实测最小 case 7.0~10.6 µs / 1~3 次发射）。⇒ 0.03 不是"难"，是贴着物理极限。

### 本轮唯一成立的改动：门控 kernel 由「每分支发一次」改为「整体发一次」

**33：impl 0.1386 → 0.0969 ms（1.43x），speedup 4.0101 → 5.7479，精度 50/50。**

新口径（`triton_dev_v2` 的 `6e4dc13c`）如实计入每次发射后暴露出来的问题：
`qkv_sel` / `gqa_attention` / `out_sel` 各按分支发 3 次，其中 2/3 不干活，
但**每次发射仍要把整个 task 空间走一遍**（`for tile in range(pid, num_tiles, num_programs)`
+ 内部标量 load + 比较）。

做法：分支权重打包成 `[nopts, D, n_max]` 连续缓冲 + 两张小表（每分支 kv 宽度、每分支 n_kv_heads），
kernel 内 `br = sel[b]` 取偏移；各分支宽度不等就按最大宽度铺 task 空间、一次标量比较跳过空 tile
（实测空转 n-tile 占 35.3%，但空转 tile 不做 GEMM，代价只有循环记账）。

### ⛔ 本轮被实测否掉的三条（都曾看似显然）

| 改动 | 结果 | 机理 |
|---|---|---|
| 80：PV 由 `p_hi/p_lo` 双原生 dot 改 **fp32 单 dot**（3→2 个 dot） | **慢 1.78 倍**（同时段 A/B：55.33 → 27.97） | fp32 dot 走**非 Cube 原生通路**，且每页多一份 `[BS,BLOCK_D]` f32 转换缓冲（64KB）。这两笔超过省下的一次 CV 握手 |
| 80：PV 改 **fp16 单 dot** | 精度 **48/50**（2 个 MERE 超阈 1.8 倍） | — |
| 21：PV 改 **fp16 单 dot** | 精度 **24/50**（7 个 fp16 精度失败 + 19 个 MLIRCompilationError） | 21 有 17 个 fp16 case，阈值 `2^-10` 比 bf16 的 `2^-7` 严 8 倍；`p` 降 fp16 的 `2^-11` 误差只剩 2 倍余量 |

⇒ **「循环内 dot 个数是成本单位」这条只在 CV 握手主导的形态（81）成立**，
在 80 上结论相反。判别信号：`aic_mac_ratio` 极低（0.02~0.05）且 `serial_ratio` 接近 2 ⇒ 握手主导；
否则算力/转换主导，此时**宁可多一个 dot 也要保原生 dtype**。

### attention kernel 的结构性上限

21 的逐 case 分解显示：**attention 只占算力的 7~14%，却吃掉约一半时间**（效率约为 GEMM 的 1/10）。
但 KV 迭代次数 ∝ `S² / (BLOCK_M × BLOCK_N)`，而 `BLOCK_M × BLOCK_N ≤ 2^20 / BLOCK_DH` 被 UB 钉死
（dh≤64 时 128×128，dh=128 时 128×64）。
⇒ **迭代次数由 UB 预算决定，不是调参能改的**——这解释了为什么 tile 全档扫描只有 3~8% 的空间。

### 测量纪律（今天两次被污染数据误导）

机器被别的会话占满时**绝对时延整体放大 6~10 倍**（framework 2.91 → 17.08 ms）。
`--lock-frequency` 只挡频率漂移，挡不住核/带宽争抢。
「取比值就好」也**不是普遍成立**：

| 算子 | torch framework | 干净 speedup | 重负载 speedup |
|---|---|---|---|
| 80（golden 发上千个小 kernel） | 2.91 ms | 59.56 | 55.33（−7%） |
| 21（golden 只有几个大 GEMM） | 0.18 ms | 2.5119 | 1.7681（**−30%**） |

⇒ 所有版本对比必须**同时段、同一块卡、背靠背连测**；开测前 `npu-smi info` 看 AICore 占用。

### ★★★ host 侧 `permute(...).contiguous()` 是达标与否的分水岭

新口径下这些**纯搬运 kernel 如实计入 impl**，而它们不产生任何计算。

**实测**：同一个算子（21），Kimi 生成版与手工版的 tile 配置**完全相同**
（fp32 `128/256/64`、低精度 `128/256/128`、`BLOCK_Q=BLOCK_KV=128`），QKV 融合投影也做对了，
唯一差别是生成版多了 **4 处 `permute(...).contiguous()`**：

```python
q = qkv_out[:, :D].view(B,S,H,dh).permute(0,2,1,3).contiguous()          # 搬运 1
k = qkv_out[:, D:D+KH*dh].view(...).permute(...).contiguous()            # 搬运 2
v = qkv_out[:, D+KH*dh:].view(...).permute(...).contiguous()             # 搬运 3
attn_flat = attn_out.permute(0,2,1,3).contiguous().view(B*S, D)          # 搬运 4
```

⇒ `impl 0.0715 → 0.1287 ms`（慢 **1.8 倍**），`speedup 2.5119 → 1.4420`，**从达标掉到不达标**。

正确做法：kernel 带 stride 直接读原始 `[M, q|k|v]` 布局（`k_base=D` / `v_base=D+d_kv`），
输出直接 `tl.store` 成 `[B, S, H, dh]`，一次搬运都不要。

**自查**：forward 里每出现一次 `.contiguous()`，都要问「这次搬运能不能用 stride 消掉」——
本类算子里答案几乎总是「能」。
