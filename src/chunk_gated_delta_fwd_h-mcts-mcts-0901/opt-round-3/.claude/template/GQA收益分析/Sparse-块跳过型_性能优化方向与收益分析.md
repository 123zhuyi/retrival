# Sparse-块跳过型 attention 算子：性能优化方向与收益分析

覆盖 4 个算子（各 50 case）：`99_BlockSparseAttnFwd` / `100_BlockSparseAttnBwd` /
`93_MsaSparseAttnFwd` / `101_SlaFwd`。
所有数字都来自 `triton-op-verifier` 的官方 `verify.py` / `benchmark.py`，
阈值与脚本未做任何修改，未使用 `--verify_not_required`。

## 0. 结果总览

| 算子 | verify | benchmark | framework(ms) | impl(ms) | **speedup（新口径）** |
|---|---|---|---|---|---|
| 99_BlockSparseAttnFwd | 50/50 | 50/50 | 661.80 | 0.0844 | **7841.98x** |
| 100_BlockSparseAttnBwd | 50/50 | 50/50 | 719.14 | 1.0769 | **667.78x** |
| 101_SlaFwd | 50/50 | 50/50 | 121.48 | 18.8983 | **6.43x** |
| 93_MsaSparseAttnFwd | 50/50 | 见逐算子报告 | — | — | — |

环境：CANN 9.1.0 runtime + triton 3.2.0 + **triton_ascend 3.2.2**（编译器用 pip 包自带的
`bishengir-compile 1.2.0`）；Ascend 910B2C，24 AIC / 48 AIV，UB 192KB。

---

## 1. ⚠️ 先说口径：两版 `benchmark.py` 的数字差两个数量级，不可比

`parse_operator_latency` 按 kernel `Name` 分组统计：

| 版本 | 做法 | 后果 |
|---|---|---|
| 旧 | 取末尾 `active_count` 行 ÷ `active_count` | **同名 kernel 一次 forward 内发射 L 次会被低估为 1/L** |
| 新（`triton_dev_v2`，「benchmark 时延改为按单次调用计算」） | 由 `整组行数 = L × active_count` 反推 L，再取末尾 `L×active_count` 行 ÷ `active_count` | 单次调用耗时正确 |

本类算子的参考实现普遍是 **B×H×块 的多重 Python 循环**，单次 forward 发射几百个 kernel
（99 实测 framework 侧 26 个 unique kernel、`launch/call = 237~425`），
而 Triton 实现是 1~6 个 kernel、每个 `launch/call = 1.0`。
⇒ 旧口径把**分子**（参考实现）压掉了两个数量级：

| 算子 | framework 旧口径 | framework 新口径 | speedup 旧 → 新（impl 代码一字未改） |
|---|---|---|---|
| 99 | 6.23 ms | 661.80 ms | 71.63 → 7841.98 |
| 100 | 5.51 ms | 719.14 ms | 5.32 → 667.78 |
| 101 | 0.44 ms | 121.48 ms | 0.024 → 6.43 |

**报 speedup 之前必须先确认 benchmark.py 的版本，并把 `launch/call` 一起报出来。**
本文所有实现的 impl 侧 `launch/call` 恒为 1.0，没有利用旧口径漏洞拆分发射刷分。

另外两条口径细节：
- `implementation.avg_latency_ms` / `framework.avg_latency_ms` 报的是**几何平均**
  （实测对照：报告值 3.0149，逐 case 算术平均 94.1179，几何平均 3.0149）。
- 编译失败的 case **不进几何平均**，所以"12 个 case 编译失败"会让 speedup **变好看**
  （100 的历史 10.13x 就是这么来的，补齐 50 case 后真实基准是 5.32x/旧口径）。
  ⇒ **看 speedup 前先看 `passed_cases == total_cases`。**

---

## 2. 头号方向：把"逐 token 的 program"提到"掩码块粒度的 program"

**这是本类算子唯一一个数量级级别的方向，实测几何平均 26.25×（旧口径 2.73 → 71.63）。**

坏味道非常好认：`grid = (total_q, H)`，kernel 里只处理一行 query，
打分写成 `tl.sum(q[None,:] * k, axis=1)`。

代价有两重：
1. **访存放大**：掩码是 128×128 块粒度的，同一个块里的 128 个 query 每人把整块 K/V 搬一遍；
   99 最大 case `total_q=1765, H=128` → 22.6 万个 program。
2. **Cube 闲置**：M=1 的"矩阵乘"退化成向量点积，且 bf16→fp32 的 cast 让 UB 流量翻倍。

改成 `(batch, q_tile=128, head)` + `grid=(min(24, tasks),)` + 核内步长循环之后：
impl 平均时延 **2.2833 ms → 0.0868 ms**，50 个 case 全部提升，最小 9.74×、最大 77.17×。

配套的 5 条（都省 UB，合起来才能把 tile 开到 128×128）：

| 手法 | 效果 |
|---|---|
| causal 上界与序列边界折叠成一个 `[BM]` 向量 `lim` | 省掉 3 个 `[BM,BN]` int32 广播临时量 |
| masked score 填 −3e38 + m 下钳 −1e30 | 省掉第二次 `tl.where(keep, p, 0)` |
| 末尾 `acc / max(l, 1e-30)` | 省掉一次 `[BM,D]` 的 `tl.where` |
| CV 重排：`tl.dot(p,v)` 提前到 `alpha`/`p_sum` 之前 | 纯重排、数值等价 |
| 块选择表放 CPU 张量 | 不产生 NPU kernel，不进 operators 统计 |

---

## 3. 选择表：建在哪、怎么建

| 放法 | 是否进 benchmark 的 operators | 何时用 | 实例 |
|---|---|---|---|
| NPU 上用 torch 算 | **进**，且常常很贵（`aclnnSort` 落到 **AiCpu**） | 永远不要 | 93 基线：`aclnnSort` 3.59ms + `aclnnRepeatInterleave` 4.15ms（全 case 累计） |
| **CPU 张量**上算完再 `.to(device)` | 不进 | 表元素数 ≲ 1e5 | 99/100：`[B, NROW, H, NCOL]`，只有几万个 bool |
| 无 dot 的 Triton kernel | 进，但很便宜 | 表大（1e6 起）/ 需按 CSR 展开 | 93：CSR → `rowsel[h_kv, block, q]` 位图 |

⚠️ 这也是 **AST 退化预检**的要求：`forward()` 里禁 Python `for`/`while` 和绝大多数 `torch.*`；
可达 helper 里只禁"主链计算算子"（`matmul`/`einsum`/`softmax`/**`cumsum`**/`masked_fill`…）。
**`cumsum` 被禁**很容易踩到——前缀和用 `torch.searchsorted` 或 `argsort` 紧致化绕开。
93 与 101 的历史交付版本都是**因为这条过不了预检**（93 用了 `torch.cumsum`、
101 的 helper 里有 `torch.softmax`/`.matmul`），本身就不是合法交付物。

---

## 4. ★ `tl.cumsum` 在 Ascend 上极慢 —— 前缀和要写成矩阵乘

同一 kernel 形状（`[11392, 2048]`）对照实测：

| 版本 | 耗时 |
|---|---|
| 带一次 `tl.cumsum` | **14.33 ms** |
| 去掉 cumsum（其余不变） | **0.254 ms** |

**56×**，而且慢到会触发 **Vector core 看门狗**（`507035`），
在正式 benchmark 里表现为跑到某个 case 直接崩。
101 的 `sla_topk_kernel` 因此占了 impl 侧 **84.87%**，`aiv_scalar_ratio = 0.8689`。

**分层 cumsum 已证伪**：拆成 `[BR, NCH, CH]` 两级，CH=64/128/256 实测 15.8 / 15.1 / 14.7 ms，
**全部不比扁平的 14.3 ms 快** ⇒ 代价与"参与前缀和的元素总数"成正比，拆轴没有意义。

**正解：把前缀和写成 `tl.dot × 下三角全 1 矩阵`，丢给 Cube。**

```python
# inclusive_prefix(x)[i] = Σ_{j<=i} x[j] = (x @ U)[i],  U[j,i] = 1 if j <= i
c = tl.arange(0, CH)
U = (c[:, None] <= c[None, :]).to(tl.float16)      # CH<=128，计数值 fp16 精确
inner = tl.dot(x, U)                                # 块内前缀：一次 Cube 乘法
tot   = tl.sum(x, 1)
pre   = tl.sum(tl.where(a[:, None] > a[None, :], tot[None, :], 0.0), 1)  # 块间前缀
pos   = inner + pre[:, None]
```
已单独验证与 `torch.cumsum` 逐位一致。
⚠️ `tl.dot` 要求 M/K/N ≥ 16：`CH` 固定 128、`NCH` 至少 16，不足的靠 mask 掩掉。

---

## 5. ⛔ 证伪方向全表（比"有效方向"更值钱）

| 方向 | 结果 |
|---|---|
| **k-centric（沿 CSR 行遍历）+ `tl.atomic_*` 跨 program 归约** | ⛔ **工具链不支持**。kernel 里只要出现 `tl.atomic_*`，凡是操作数由本 kernel 内 Vector 运算（`exp`/`.to()`）产生的 `tl.dot` **就静默返回全 0**；两个操作数都来自 `tl.load` 时正常。表征是 WARNING `AutoBlockify disabled ... Unsafe ops: atomic operations`。⇒ **split-K / flash-decoding 式的跨 program 原子归约在 Ascend Triton 上暂时不可用** |
| ↑ 的全部绕法（13 个对照实验） | `tl.debug_barrier()` 无效；`has_auto_blockify_blacklist_op=False`（编译器自己给的建议）无效；关 `multibuffer`/`enable_ubuf_saving` **更糟**（连 `atomic_max` 求出的 m 都错）；转置形式 `tl.dot(vT, tl.trans(p))` **运行期 AI Core aivec trap**；p 经 per-program 小 scratch 走一遍 GM 再 load **编译报错**；atomic 换成 `tl.store` **bishengir SIGABRT** |
| 逐 token 压缩表（每 token 只扫自己的 topK 块） | ⛔ **全面变差**：case 20 从 2.78 → 21.3 ms、case 40 从 54.5 → 954 ms。任务数从 3072 暴涨到 98304，tile 缩到 `[2,128]` ⇒ 每任务标量前导占满。**访存量不是唯一目标函数** |
| 去掉逐块的数据依赖分支 `if tl.max(sel) > 0:` | ⛔ 无收益（1472 vs 1386 ms，略慢）。曾假设它打断流水，**证伪** |
| 分层 cumsum | ⛔ 见 §4 |
| `tl.atomic_add` 抢占式紧致化（替代 cumsum） | 语义**正确**（cnt 与集合都对），但同址 lane 高度竞争，**触发 Vector core 看门狗**，不可用 |
| 在 LUT kernel 内用 `tl.sum(...)` 全量归约重算 n_gt | ⛔ 引入每行一次 向量→标量 往返，NROW 上万时直接 **aicore timeout(507014)** |
| `enable_ubuf_saving` 编译开关 | 在 bishengir **0.1.0 上根本不可用**（`Unknown command line argument`，编译直接失败）；1.2.0 上可用且是 128×128 能编过的前提。**开关的可用性先于收益** |
| `BM=BN=128` 用于 paged topK 形态 | ⛔ `ub overflow`（编译器已自动关 code-motion + multi-buffer 重试仍失败）。同一个 128×128 在 block-mask 形态上却能编过 ⇒ **可编域是"形态 × 编译器"的函数，不能跨算子照抄** |

---

## 6. 精度：判定顺序错了会把 bug 归错类

1. **先看覆盖率**。选择表行粒度 ≠ kernel `BLOCK_M` 时，输出只有一部分被写，
   其余留在 `torch.empty` 的未初始化内存里。实测形态：
   `matched_ratio=0.558 / max_abs_diff=5.0 / MERE=0.43` **+ NaN 位置断言同时触发** ——
   看起来完全像"精度不够"。
2. **再看选块集合**。选块是离散的，差一个块 = 整行结果不同，
   表现为 `matched_ratio` 断崖而不是缓慢劣化。
   101 上两个必须逐位对齐的点：**`aclnnTopk` 并列取小索引**（17 case 逐个比对确认）、
   **`-0.0` 必须归一到 `+0.0`**（CPU 穷举 3120 组：不修 179 组不一致，修完 0 组）。
3. **最后才是数值精度**。

### `tl.dot` dtype 契约在本类算子上的落点

| 量 | 是否需要提精度 | 理由 |
|---|---|---|
| q / k / v / dout | **否** | 本来就是 bf16/fp16 存储，升 fp32 一位有效数字都不增加，却让 UB 流量翻倍、Cube 走 fp32 通路 |
| `p`（softmax 输出） | **是**，hi+lo 二段拆分 | 真 fp32 量，误差被 `Σp|v|/|Σpv|` **放大约 20×**（实测 rel 2~4%，bf16 阈值 7.8e-3）。修完 MERE 0.43 → 1e-6 |
| `dS`（反向） | 同 `p` | — |

⚠️ **fp16 上二段拆分不是无条件安全的**：实测反向的 `P` 可低到 **4e-21**，
远低于 fp16 最小非规格化数（≈6e-8），`ph = P.to(fp16)` 直接变 0。
**拆分前先看动态范围**；bf16（最小非规格化数 ≈9e-41）不受影响。

### 定位手法：dump kernel 中间量，不要只看最终输出

93 的 k-centric 版本最终输出全错，但把 `m` / `l` / `acc` 单独 dump 出来跟 torch 复算比，
立刻看到 **`m` 完全正确、`l` 完全正确、只有 `acc` 恒 0** —— 一眼锁定是 PV 那一步，
省掉在 tiling / 精度上的全部无效搜索。

---

## 7. 环境是第一变量

1. **非交互 shell 读不到 CANN `set_env.sh`**：`~/.bashrc` 开头的 `[ -z "$PS1" ] && return`
   会让脚本静默落到旧 CANN + 旧 bishengir，**所有编译器相关结论整片作废**。
   本轮前半程就踩了这个：在 bishengir 0.1.0 下扫出的 tile 可行域
   （D=128 只有 32×32）换到 1.2.0 后变成 128×128 全通，**16 倍面积差**。
2. **编译器来自 pip 包，不是 CANN**：`triton_ascend` 的 `_get_npucompiler_path()`
   优先用 `backends/ascend/bishengir/bin/bishengir-compile`，找不到才回落 `PATH`。
   ⇒ **CANN 的 set_env 决定 runtime，pip 包决定编译器**，两者可以不同源。
3. **性能测量必须独占 NPU**：同一张卡上并行跑调试脚本，会让 benchmark 的大 case
   报 `Vector core execution timed out`，失败形态伪装成算子问题（100 实测踩过）。
4. **环境探针**：torch 参考实现代码恒定，其 `framework.avg_latency_ms` 是天然探针，
   设基准带、超出即判该次测量无效并重测。**比值型指标不能自证有效。**

---

## 8. 天花板估算：先算账再动手

块稀疏算子最容易高估收益。三步：

1. **块加载次数** = `Σ_tiles(每 tile 扫的块数) × heads × batches × 块字节数`；
2. **工作集** = `nblocks × 块字节数`，若 ≤ L2 则上面那个是 **L2 流量**不是 HBM 流量；
3. **每次块迭代的理论代价** = Cube MAC/200TFLOPS + Vector 元素数/1e13 + 访存/带宽。

一次实测对照：93 的 case 40 有 147456 次子迭代、实测 54.5 ms ⇒ **每次 8.9 µs**，
而理论只需 ~12 ns，**差 700 倍且与访存量无关**
⇒ 判定为"指令发射/同步开销主导"，此时**继续减少访存量没有用**，
只能把每次迭代做的事变大（加大 tile、去掉子块拆分、砍每次迭代的 Vector 趟数）。

### 逐 tile 扫描 vs 逐 token 压缩表的判据

设 `n_vis` = 该 q-tile 的因果可见块数，`topK` = 单 token 选中块数，`BM` = tile 行数，
一个 tile 要扫的块数 ≈ **`min(topK × BM, n_vis)`**：

- `topK × BM ≥ n_vis` ⇒ 并集≈全部可见块，两种写法等价，**只能靠加大 BM 摊薄**；
- `topK × BM < n_vis` ⇒ 压缩表严格更省。

⚠️ 但压缩表把任务粒度缩到 `[G, D]` 时，**tile 太小的代价会吃掉全部访存收益**（§5 已证伪）。
