# 81_GqaSparseDecode 性能优化方向与收益分析

> 面向「块稀疏 GQA decode」这一类算子（vLLM MiniMax-M3 `_gqa_sparse_decode_kernel` 族）
> 在 Triton-Ascend / 910B2C 上的可迁移经验。状态：迭代进行中，随轮次更新。

---

## 0. 算子形状与工作量画像（先算账）

KV cache 布局 `[num_blocks, num_kv_heads, 128, 2*head_dim]`，K = 前 head_dim、V = 后 head_dim；
稀疏块 128 == 一页。`topk_idx [KH, total_q, topk]` 给出每个 (kv_head, query token) 选中的页。

50 个 case 的参数空间：

| 维度 | 取值 |
|---|---|
| dtype | fp16（9 个 case）/ bf16（41 个） |
| head_dim D | 64（6 个：case 1/2/3/5/13/15）/ 128（44 个） |
| G = H/KH | 4 / 6 / 8 / 16 |
| topk | 4 / 8 / 16 |
| rtopk = min(topk, ceil(kv_len/128)) | **恒为 {1,2,4,8,16}** |
| total_q × KH（= program 数） | 2 ~ 256，**其中约 20 个 case < 24（欠并行）** |

**工作量 = Σ(t,kh) rtopk(t) = 18398 页**，每页读 `128×2D×itemsize`（D=128/bf16 即 64 KB），
合计 1.18 GB，**没有冗余**：不同 (t,kh) 选中的页互不重复。

耗时模型（实测反推，误差 <10%）：

```
kernel_us ≈ 每页代价 × ceil(programs / 24) × 每 program 页数
```

**固定启动开销几乎为 0**（case 5 只有 1 页，耗时 6.7us ≈ 1 页的代价）。

---

## 1. 第一杠杆：任务单元必须是 (token, kv_head)，不是 (token, q_head)

**收益：0.2233 → 1.3791（6.18x），是本算子迄今唯一的量级级改动。**

原始实现按 `head_idx in range(pid, total_q*H, num_cores)` 分任务，`kh = h // g_size`。
于是同一个 kv_head 组内的 G 个 q head **各自把同一份 `[128, 2D]` KV 页重新读一遍**。
主导耗时的 case 37~45 全部是 G=16 ⇒ **KV 访存被放大 16 倍**。

改成一个 program 处理整组 G 个 q head（q tile `[BLOCK_G, D]`）后：

- KV 访存降到 1/G；
- QK 段 `tl.dot` 的 M 从 1 变成 BLOCK_G=16，正好是 Cube 的 fractal 粒度；
- PV 段从 `tl.sum(p[:,None] * v, axis=0)` 的**向量广播归约**（[128,1]×[128,D] 全在 Vector 上）
  变成一次 `tl.dot`，从 Vector 挪到空闲的 Cube。

> **可迁移判据**：任何 GQA/MQA 算子，只要 kernel 里出现 `kh = h // G` 且循环变量是 q head，
> 就一定在按 G 倍重复读 KV。这是 decode 场景最容易被漏掉的一处放大。

---

## 2. 成本单位是「循环内 dot 的个数」，不是 tile 的大小

这是本算子最反直觉、也最有指导价值的一条。

`diag/probe_iter_cost.py`（96 program × 16 页的合成 kernel）逐段加料：

| 内容 | us/页 |
|---|---|
| 只算地址 | 0.80 |
| + load K | 0.62 |
| + load K,V + QK dot | 1.54 |
| + softmax 向量段 | 1.50 |
| **+ PV dot** | **8.85 ~ 10.56** |
| PV 的 p 换成**循环不变量** | 3.06 |
| 两个 dot、操作数全来自 GM（`probe_dma.py` MODE 1） | 1.69 |

**结论：dot 本身很便宜（两个 dot + 两条 DMA 只要 1.69us/页）；贵的是「dot 消费了向量段刚产出的操作数」。**
FlashAttention 单次迭代的依赖链 `Cube: QKᵀ → Vector: softmax → Cube: PV → Vector: acc更新`
天然要求每页一次跨核往返，这就是 5~6 us/页 的来源。

实证（两条独立方向都验证了这条）：

- **v4**：PV 从 1 个 fp32 dot 换成 2 个原生 dtype dot（精度更好），
  循环内 dot 2→3，实测时间 **正好翻倍**（5087 → 10500 us，speedup 1.3791 → 0.7089）。
- **v5**：把 NP 个页合并成一个大 tile，dot 数从 2/页 降到 3/(NP 页)，
  逐 case 看 NP 越大比值越好（NP=1 → 0.52、NP=2 → 0.5~0.87、NP=4 → 0.74~0.95）。

> 与 28_FlashAttentionFwd 的「每次 kv 迭代 4~6us 固定握手成本，几乎与
> BLOCK_Q/BLOCK_KV/BLOCK_D/dtype 无关」是同一现象的两种测法。
> **两个算子独立复现 ⇒ 这是 910B2C + Triton-Ascend 的平台特征，不是某个 kernel 的偶然。**

### 2.1 推论：摊薄握手的正确做法与错误做法

- ✅ **正确**：增大单次迭代覆盖的 KV 行数（本算子即「一次处理 NP 个 topk 页」）。
- ❌ **错误**：用「行基址向量 gather」把 NP 个不连续页拼成一个 tile。
  实测 `aiv_mte2_ratio` 从 0.081 暴涨到 0.539 —— **gather 形态的 load 走不了 Cube 的
  GM→L1 直载，被迫由 AIV 搬进 UB 再转交给 Cube**，省下的握手被多出来的搬运吃光。
  这条是本算子踩到的最贵的坑。

---

## 3. DMA 形态：两次跨步半行 vs 一次连续整页

KV cache 一行是 `[K(D) | V(D)]`。分两次 load（K 取前 256B、V 取后 256B，行距 512B）
与一次连续 load 整行，实测（`diag/probe_dma.py`，全部喂 Cube 不经 Vector）：

| load 形态 | us/页 | 有效带宽 |
|---|---|---|
| 只读 K | 0.575 | 1367 GB/s |
| **读 K + V（两次跨步半行）** | **1.692** | 929 GB/s |
| **一次连续读整页 `[BS, 2D]`** | **0.642** | 2452 GB/s |

同样的数据量，拆成两条跨步 DMA 比一条连续 DMA 贵 **2.6 倍**。

**怎么用一条 load 还能分别拿到 K 和 V**（不需要 `tl.split` / 三维转置）：

- QK：把 q 补零成 `[BLOCK_G, 2D]`（后 D 列恒 0），则 `dot(q2, kvᵀ) == q·Kᵀ`，
  多出来的 K 维乘的都是 0；
- PV：`dot(p, kv)` 得到 `[BLOCK_G, 2D]`，前 D 列是 p·K（丢弃），后 D 列正是 p·V。
  `acc` 保持 `[BLOCK_G, 2D]`，只在最后 store 时用**列掩码 + 指针偏移 `-D`** 取后 D 列，
  被掩掉的 lane 不写内存。
- 代价：QK dot 的 K 维和 PV dot 的 N 维都翻倍。算术强度 16 flop/byte ≪ 机器平衡点 ~235
  ⇒ **Cube 本来就闲，拿 Cube 换 DMA 是划算的。**
- 约束：要求 `BLOCK_D == D`（整页宽度 `2*BLOCK_D` 必须正好等于 2D）。

---

## 4. 精度：PV 段用什么 dot

golden 是 `softmax(fp32) @ V.float()`，输出再 `.to(q.dtype)`。判据是三项 AND：
allclose（atol 9e-2/1e-1，很松）、matched_ratio ≥ 0.9、**MERE < 2^-10(fp16) / 2^-7(bf16)**。

| 方案 | 循环内 dot 数 | 精度（case 42，最严） | 速度 |
|---|---|---|---|
| `dot(p_f32, v.to(f32))` | 2 | 通过（MERE ~1e-5） | **最好（1.3791）** |
| `dot(p.to(v.dtype), v)` 单次原生 | 2 | **FAIL，MERE 9.004e-3 > 7.812e-3** | — |
| 「分子分母口径一致」（l_i 也用 p̃） | 2 | **FAIL，MERE 1.30e-2** | — |
| `dot(p_hi,v) + dot(p_lo,v)` 双原生 | 3 | 极好（MERE 1.03e-5） | 0.7089（**慢一倍**） |

要点：

- **p 舍入到 bf16 的半 ulp 2^-9 在 `Σp·v / Σp` 上会被放大越阈**，与 80_GqaSparseFwd 的结论一致；
- 但 80 号那套 `p_hi/p_lo` 在本算子上是**性能负项**，因为本算子的瓶颈是 dot 个数而不是精度余量；
- ⚠️ BRIEF 记的「`.to(dtype).to(fp32)` round-trip 会被编译器消除、必须 bitcast+RNE」在
  bishengir 1.2.0 上**没有复现**：最朴素的
  `p_hi = p.to(dt); p_lo = (p - p_hi.to(tl.float32)).to(dt)` 就把 MERE 从 9e-3 压到 1e-5。
- **先用 CPU 模拟筛精度方案**（`diag/sim_prec.py`）：不占 NPU、秒级出 50 个 case 的三项判定，
  「口径一致」这个方案就是这么在不写 kernel 的情况下被否掉的。

---

## 5. 编译器陷阱：fp32 `tl.dot` + N=64 + 运行期 trip-count 循环

**症状**：D=64 的 6 个 case 整组输出 NaN，且**只在偶数列**，奇数列是 0~0.8 的垃圾正数；
D=128 的 44 个 case 全过。同一份代码在旧工具链（bishengir 0.1.0）上 50/50 全过。

**判据**：`tl.dot` 满足以下三条时被 bishengir 1.2.0 误编译 ——
1. 操作数是 fp32；
2. 输出宽度 N == 64；
3. 位于运行期 trip-count 的 `scf.for` 内。

打破任意一条即规避：`BLOCK_D` 抬到 128（N=128）、改原生 dtype dot、或把 trip 改成常量。

**这条修正了 BRIEF §4.2**：「本机 fp32 `tl.dot` 可用且与 torch fp32 matmul 逐位一致」
的探针是**循环外**测的，适用条件应收窄为「循环外 或 N ≥ 128」。

**同时证伪了 BRIEF §4.3 的首选假设**（「K/V 同时无掩码 load 触发 bishengir 崩溃」）：
16 种孤立组合全部正常；在真 kernel 上换成 80 号已验证的「K 掩码+where / V 无掩码」写法仍然 NaN。
**80 号算子的编译器结论不能直接搬到 81 号。**

> **可迁移方法**：低精度输出的 NaN/垃圾，先看**列的分布**。
> 「恰好一半、且是偶数列」是 tile 交织排布被写坏的指纹，直接指向 dot 的输出宽度，
> 而不是掩码、边界或 online softmax 的 `-inf`。

---

## 6. 并行度：本算子约 20 个 case 的 program 数 < 24

`programs = total_q × KH`，取值 2 ~ 256。case 1/4/5 只有 2 个 program（24 个核只用了 2 个）。

但**低并行度不是当前瓶颈**：耗时按页线性，欠并行的 case 页数也少
（case 5 只有 1 页 → 6.7us，已是 9.19x）。真正吃亏的是页数多的大 case。

**split-K（flash-decoding）先算账再动手**：`benchmark.py` 按 kernel 名 groupby，
多一个**不同名** merge kernel 会给**每个 case** 都加一次它的单次平均时延（估 5~8us）；
收益只在 programs<24 的约 20 个 case 上，其余 30 个 case 净亏。
⇒ 只有当每页代价压到 ~1us、欠并行成为主导项时才值得重新评估。

---

## 7. 与其他算子/历史结论的差异（历史结论确实会作废）

| 条目 | 历史结论 | 本算子实测 |
|---|---|---|
| K/V 同时无掩码 load | 80_GqaSparseFwd：触发 bishengir 崩溃，必须 K 走掩码 | **证伪**，与本算子的 D=64 NaN 无关 |
| fp32 `tl.dot` | BRIEF §4.2：可用且逐位一致 | **需加条件**：循环内且 N=64 时误编译 |
| `p_hi/p_lo` 二段拆分 | 80_GqaSparseFwd：精度必需；28_FlashAttentionFwd：+2.7% | 精度确实必需（若走原生 dot），但**性能 -49%**：本算子瓶颈是 dot 个数 |
| `.to(dtype).to(f32)` round-trip 被编译器消除 | 80 号：必须 bitcast+RNE 规避 | **未复现**（bishengir 1.2.0），朴素写法即可 |
| `aic_scalar_ratio` / `aiv_scalar_ratio` 高 | MQA/28：是同步等待被计入标量管线，不是标量运算多 | **一致** |
| 每次 KV 迭代的固定握手成本 | 28_FlashAttentionFwd：4~6us，与 tile 大小无关 | **一致**：本算子 5~6us/页 |
| 放大 tile 摊薄握手 | 28：被 UB 卡在 BLOCK_Q×BLOCK_KV≈8192 | 本算子 BLOCK_G×BS=2048，**有空间**，但页不连续，合并方式受限 |

---

## 8. 后续轮次补充（v7 / v8 / v9）与**测量口径的决定性变更**

### 8.1 ⛔ 又证伪三条（合并进第 7 节的「历史结论会作废」一起看）

| 方向 | 结论 | 证据 |
|---|---|---|
| **PV dot 错开一轮，打断 Cube→Vector→Cube 依赖** | **1.3791 → 1.2647，证伪** | `serial_ratio` 1.738 → 1.707 **几乎没动** ⇒ CV 握手是**按 dot 结构性插入**的，与操作数来自哪一轮无关。错位记账本身没错（CPU 上 200 组随机用例对拍 v7 == v2 == 一次性 softmax，误差 <1e-12），精度也确实 50/50 |
| **PV dot 由 fp32 换 fp16 原生** | **1.3088，持平**（但**精度通过**，且省 32KB UB） | dot 个数没变 ⇒ 时间不变，再次印证「dot 个数是成本单位」。它的价值只在腾 UB |
| **整页连续 load**（DMA 1.692 → 0.642 us/页） | **更慢，证伪** | 干净重测 impl 0.0956 vs v2 的 0.0750 ms。省下的 DMA 抵不过两个 dot 的 K 维/N 维翻倍（`acc` 也从 `[16,128]` 变 `[16,256]`） |
| **多页合并成一个大 tile**（NP≥2） | **UB 墙，三种写法全部编不过** | 标量构造行基址 / `tl.cat` / `tl.join`+3D `tl.trans`，全部 MLIR `PlanMemory Failed`。记账：基准已占 **188KB / 192KB** |
| **转置 QK 形态**（`dot(k, qᵀ)` 省掉 `trans(k)`） | **慢一倍，证伪** | 11.9 vs 5.78 us/页 |
| **利用 token 之间的 topk 页复用** | **账上不成立** | 逐 case 统计：合计页读 18398 / 去重后 17310 = **冗余仅 1.06x**（只有 3 个 dql>1 且「全选」的 case 有 2x） |

### 8.2 成本模型（两点联立解出，误差 <10%）

```
kernel_us ≈ (5.81 us/页 × 总页数 + 0.71 us/program × program 数) / 24 核
```

其中 5.81 的构成：**两个 dot 的固定 CV 握手 ≈ 4.1 us + DMA 1.69 us + MAC ≈ 0.05 us**。
⇒ 即使把 DMA 全部归零也只有 4.1 us/页。**这个账要在动手前算**，它一次性关掉了所有访存方向。

### 8.3 UB 记账（本算子一切扩张撞墙的唯一原因）

910B2C 单核 UB **192KB**，v2 的占用：

```
k[128,128]bf16 32 + tl.trans(k) 32 + v[128,128]bf16 32 + v.to(f32) 64
+ q 4 + scores 8 + p 8 + acc 8  =  188KB / 192KB
```

**只剩 4KB。** 记账时**必须把 `tl.trans(x)` 的物化算进去**——它会额外占一份与 `x` 等大的空间，
这是最容易漏的一项。

### 8.4 ★★ 测量口径变更：同一份代码 1.3791 → **41.91**

`triton_dev_v2` 的 `6e4dc13c fix(triton-op-verifier): benchmark 时延改为按单次调用计算，
支持 forward 内多次启动 kernel`：由整组行数反推发射次数 `L = len(group)/active_count`，
取末尾 `L×active_count` 行求和再除以 `active_count`
⇒ **一次 forward 内的多次发射如实全部计入**。

| 口径 | framework | impl | speedup |
|---|---|---|---|
| 旧（按 kernel 名 groupby，只计一次） | 0.1044 ms | 0.0757 ms | 1.3791 |
| **新（按单次调用）** | **3.1449 ms** | 0.0750 ms | **41.9099** |

**impl 侧几乎不变**（本算子只有 1 个 kernel、只发射 1 次），**变的全是 torch 那一侧**——
torch 参考是 `for pid_b in range(total_q)` + `for kh in range(KH)` 的双层 Python 循环，
一次 forward 发射上千个小 kernel，旧口径把它的真实代价低估了约 **30 倍**。

⇒ 两条必须同时记住：
1. **第 6 节「split-K 会给每个 case 加一次 merge kernel 时延」那条在新口径下依然成立**，
   但「把不同名 kernel 合并成同名能刷分」的漏洞已经堵上，**新口径下按真实 wall-clock 设计即可**；
2. **跨版本比较历史分数前必须先核对用的是哪一版脚本**。本文件第 0~7 节的所有 speedup
   都是旧口径下的相对值，用来做**方案之间的横向比较**仍然有效
   （impl 绝对时间没变），但**不能与新口径的绝对分数混用**。

### 8.5 共享机器上的测量纪律（差点让一个负优化被固化）

v9 第一次测得 1.6007（看起来比 v2 的 1.3791 好），实为污染数据：当时另一会话在跑
`msopprof simulator` + 另一个 verify，NPU 0 的 AICore 占用 93%，
framework 从 0.105 ms 飙到 0.367 ms、impl 从 0.076 飙到 0.230，**绝对值整体放大 3 倍**。
换到空闲卡 + `--lock-frequency` 重测后结论直接反转。

⇒ **三道闸门缺一不可**：全局互斥锁、必须测满全部 case、工具链指纹前后对照；
再加上 `--lock-frequency` 与**开测前 `npu-smi info` 看一眼 AICore 占用**。

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
