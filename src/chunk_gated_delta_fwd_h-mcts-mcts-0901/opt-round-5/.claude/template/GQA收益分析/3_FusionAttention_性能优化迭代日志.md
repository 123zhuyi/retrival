# 3_FusionAttention 迭代日志（0817/3，2026-08-17 单日攻坚）

目标：61/61 精度全过前提下 speedup_vs_torch 最大化。
环境：Ascend 910B（`triton_ascend_320` conda env）/ triton 3.2.0 / triton_ascend 3.2.2 / CANN 9.1.0。
同机对照基线：0810/3 `attempt_from_old`（旧环境 opt_iter_15 迁移版，**0.8914x**，61/61）——
同 golden 任务（61 case 配置 JSON diff=0 行，CANN dispatch 逐 case 相同，framework 侧延迟
61 case 一致 ±4.5%），两版数字可直接横比。

## 时间线总览（当日 8 轮 bench，全部 61/61）

```
17:45    18:19     18:38     18:57     19:12    19:29     19:43     19:52
0.2217 → 0.3523 → 0.7675 → 0.7976 → 0.5849 → 0.7644 → 0.7727 → 0.8068   （×3.6）
bench    bench2    bench3    bench4    bench5   bench6    bench7    bench8
```

impl 平均延迟 0.1241 → 0.0337 ms（**−73%**）；framework 恒 0.027 ms。

精度线（11 轮 verify）：16:19 **58/61**（3 失败）→ 16:26 58/61 → 17:09 60/61 →
17:33 **61/61 首过** → 17:56 60/61（case 60 回归）→ 18:06 起 61/61 稳定。

---

## 第一阶段：精度攻关——golden 是 CANN npu_fusion_attention 的真实行为，不是数学语义

### 15:58 语义探针（probe_f7.py / probe_precision.py）

先探 golden 行为再写 kernel，两个探针决定后续全部架构：

- **probe_precision.py**：对照 CANN 输出 vs 手工公式（fp16 matmul / fp32 matmul /
  fp16-round 中间量），用 verify 同款三项阈值（fp16 rel=2^-10、bf16 2^-7、fp32 2^-13）。
  结论：CANN 的 QK/PV 精度路径随 dtype/inner_precise 分档，不是单一公式。
- **probe_f7.py**：带真实值域输入（normal(μ±5, σ0.1~2)，不用裸 randn），逐语义项探测
  sink / sparse_mode 窗口 / pse 广播 / workspace 形状。

### 三个精度根因（61/61 的全部来源）

1. **online rescaling 与 CANN 两趟结构不一致**：CANN golden 实际是"先全局 max 再 exp"
   的两趟结构（workspace 物化路线），triton 经典 online softmax（逐块 rescale）在
   临界 case 上 MERE 超阈。修法：online 路径只在 scale==1 且无 attrs 时启用
   （此时逐位差被压到阈内），其余走物化两趟。
2. **D=512 的 chunked QK 只能三段式**：D=512 时整行 K 放不进 UB/cbuf，triton 融合 dot
   的分块归约顺序与 CANN 单趟全宽 dot 不同 ⇒ 必须"QK 物化（分半归约）→ softmax 读
   workspace → PV"三段。`t_2half.py`/`t_chunkdot.py`/`t_dot32.py` 扫过全部归约写法，
   只有显式两半 + workspace 物化过阈。
3. **workspace 4-stride 对齐**：softmax_max/sum 分配成 `[B,N,S,8]` fp32（8 路 stride），
   复刻 CANN workspace 布局。softmax kernel 对该布局的读写方式错一个 stride 就
   溢出/错值（case 60 在 17:56 的回归即此类）。

### 16:19–17:33 从 58/61 到 61/61

| 轮 | 结果 | 失败 case | 根因 |
|---|---|---|---|
| 16:19 verify | 58/61 | 55 等 3 个 | MERE 1.03e-3 > 9.77e-4（fp16 临界超阈），matched_ratio 0.965 本身够 |
| 17:09 verify3 | 60/61 | 40（fp32） | max_error_cap 8/8192 违例，1.27e-3 > atol 1e-3——fp32 档输出舍入路径错 |
| 17:33 verify4 | **61/61** | — | 两趟结构 + dtype 分档舍入落地 |
| 17:56 verify5 | 60/61 | 60（pse 1row） | softmax_out 对 pse_1row 的 s-stride 写错（matched_ratio 0.0018 断崖式全错） |
| 18:06 起 | 61/61 | — | stride 修复，此后 6 轮 verify 全绿 |

**判据沉淀**（与家族经验一致）：精度失败先看失败形态——`matched_ratio` 断崖 ⇒ 寻址/stride
bug（去查 layout 分支）；MERE 临界超阈 ⇒ 归约顺序/舍入路径（去对齐 CANN 的两趟结构）；
max_error_cap 个位数违例 ⇒ 边界 dtype 舍入（分档处理）。

---

## 第二阶段：性能爬坡——三轮结构 + 路由收敛

### 17:45 bench1 = 0.2217（首过精度的直接产物）

架构：全部 case 走保守物化路线（score_kernel 物化 → softmax → output），无路由。
impl 0.1241 ms，是 framework（0.0275）的 4.5 倍——精度优先的代价。

### 18:19 bench2 = 0.3523：flash 两趟 kernel（probe_twopass.py 验证）

`probe_twopass.py`（17:03）验证两趟 flash（per-tile max/exp/sum + 终趟 PV）在无 attrs、
D≤128 case 上与 CANN 精度兼容 ⇒ 引入 `_fa_kernel` 两趟路径，plain case 换装。
大 case 仍慢（case 39 0.07x 级），但 geomean +59%。

### 18:38 bench3 = 0.7675：online 单趟成为主力（t_online_*.py 系列）

18:39–18:40 的 `t_online_native.py`/`t_online_all.py` 是决定性实验：**online 单趟 +
native dtype dot（不升 fp32）**在 scale==1、无 attrs、D≤128 时与 CANN 逐位兼容。
⇒ 新增 `_fa_online_kernel`（单趟、QK 与 PV 共享一次 K/V 扫描、BQ=BKV=128 大 tile），
路由优先级 online > 三段式 > 两趟。这是当日最大单笔（+118%）。

### 18:57 bench4 = 0.7976：三段式 BLOCK 调优 + D>512 分半归约

`t_3stage_block.py` 对 case 55（BNSD [4,32,512,64]）打点：三段式 BD≤128 档
BQ,BKV 从 64,64 试到 128,128，UB 压线即回退。`_score_half_kernel` 对 D>512
拆两半归约（HALF=0/1 两次发射），QK 物化只算一次。

### 19:12 bench5 = 0.5849（回退轮）：BLOCK 上探过猛

三段式 BQ=128 企图扩大到 pse/mask case，UB 溢出 + 编译退化，16 个 3stage case 集体变慢。
**教训**：本轮没有 verify 前置就直接 bench，回退后从 19:29 起恢复"每改一处先 verify 再 bench"。

### 19:29–19:52 bench6→8 = 0.7644 → 0.7727 → 0.8068：路由收紧 + online 收尾

- online 启用条件收紧为 `d_q≤128 ∧ 无pse ∧ 无sink ∧ 无mask`（避免临界精度回退）；
- `t_online_scale.py`（19:31）确认 scale≠1 的 case 从 online 摘出回三段式（scale 进
  softmax 前乘 vs 后乘的舍入差在 fp16 档超阈）；
- 终态路由：**online 40 case / 三段式 21 case / 两趟 0 case**（`_fa_kernel` 保留为
  workspace>256MB 兜底，61 case 内未触发）。

---

## 终态架构（700 行，4 个 kernel + 三路路由）

```
forward
 ├─ online（40 case）：_fa_online_kernel 单趟 flash，BQ=BKV=128，persistent grid
 │    条件 d_q≤128 ∧ scale==1(缺省) ∧ 无 pse/sink/mask
 ├─ 三段式（21 case）：_score_half_kernel（D≤128 单发；D>512 HALF=0/1 两发分半归约）
 │    + _softmax_out_kernel（attrs 全支持：pse 1row/广播、bool/uint8 mask、sink、
 │    sparse_mode 窗口、GQA group 寻址、fp16/bf16/fp32 输出分档）
 │    条件 B·N·S_Q·S_K·4B ≤ 256MB workspace
 └─ 两趟兜底（0 case）：_fa_kernel，workspace 超大时
```

4 种 layout（BSH/SBH/BSND/BNSD）全部纯 stride 寻址，零转换拷贝；
smax/ssum 按 CANN `[B,N,S,8]` 4-stride workspace 语义分配。

## 逐段战绩（route × attrs × D 分组，new vs old 逐 case）

| 组 | n | old geomean | new geomean | 胜者 |
|---|---|---|---|---|
| online, plain, D=64 | 11 | 0.979 | 0.832 | OLD |
| online, plain, D=128 | 29 | 0.855 | 0.848 | OLD（微） |
| 3stage, attrs, D=128 | 16 | 0.923 | **0.997** | **NEW** |
| 3stage, plain, D=512 | 5 | **0.823** | 0.287 | **OLD** |

- **新版净赢面**：attrs case（pse/sink/uint8-mask）——case 58（uint8 mask）
  0.124→1.274（**10x**）、case 30/46/47/61 0.81→1.24~1.34、case 53 1.02→1.55；
  小 shape case 57 3.54→4.66。例外：case 60（pse 1row）新版三段式 50.6µs 反而慢于
  old 21.1µs——`_softmax_out_kernel` 47.2µs 占 93%，pse_1row 的 S_Q 维 stride=0
  广播寻址没做特化，是三段式 attrs 路径的下一刀。
- **新版净输面**：
  - **D=512 GQA 组**（case 38/40/41/49/39 = BSH q[·,128,4096] hn=8 ⇒ D=512，
    k/v D=128）：新版三段式 `_score_half_kernel`×2（30.2µs×2）+ `_softmax_out_kernel`
    （35.1µs）≈ 95µs，old 的 `score_kernel_flash`（BLOCK_D 内层循环、BLOCK_M=128/
    BLOCK_N=64）整体 28µs —— **慢 3.4 倍**。old 赢在 QK 段直接大 tile 内层 D 循环
    （BLOCK_D=64），一次发射吃掉 512 维；新版为对齐 CANN 归约顺序拆成两半 × 两次发射，
    launch + workspace 往返翻倍。
  - case 3（[2,512,4096] D=128 大 S²）、case 25/32/39（大 S₂）大序列 online/三段式
    均饥饿：KV 循环串行，0.07~0.44x。

---

## 与 0810 attempt_from_old（0.8914x）的关系

两版是**两代不同方案**，不是同一方案的迭代：

| | old（0810，opt_iter_15 迁移） | new（0817） |
|---|---|---|
| 结构 | 多路径 dispatch：单 kernel flash（workload≤10000）/ score_softmax 融合 / 三件套 | 三路路由：online 单趟 / 三段式 / 两趟兜底 |
| kernel 数 | 10 个（score×2/softmax/score_softmax/dropout/output×3/flash…） | 4 个 |
| 调参 | 15 轮 opt_iter 手调（BLOCK_M=256/BLOCK_S 自适应…） | 单日 8 轮 bench |
| D=512 GQA | `score_kernel_flash` BLOCK_M=128/BLOCK_N=64 内层 BLOCK_D 循环，28µs | 分半归约两发 + softmax，95µs |
| uint8 mask / pse 1row | case 58 崩 0.124x；pse 1row stride 有 bug（case 60 老 0.967→新已修） | 1.274x / 已修 |
| sink | CANN 9.1 UB 对齐异常 → 禁融合路径绕开 | kernel 原生支持 |
| 布局 | `_to_bnsd` view 转换 + stride | 纯 stride，零转换 |

**合成上限**：逐 case 取两版最大值 = geomean **1.0184x**（61 case 中 new 赢 31）。
⇒ 后续最优路线 = old 为主体（0.8914），移植 new 的三处修复（uint8 mask 分支、
pse 1row stride、sink 支持），再把 old 的 `score_kernel_flash` 大 D 路径按 new 的
精度结论（分半归约顺序）校准——这是超越 0.8914 的最短路径。

## 已证伪/回退方向（当日实测）

| 方向 | 结果 | 证据 |
|---|---|---|
| online 无条件铺开（含 scale≠1 / 有 attrs） | 精度回退 | t_online_scale：scale 前乘/后乘舍入差 fp16 档超阈；attrs case online 无 workspace 语义 |
| 三段式 BQ,BKV 上探 128,128 全档 | 0.7976→0.5849（−27%） | bench5：UB 溢出 + 编译退化，pse/mask case 集体变慢 |
| D=512 融合 chunked dot（单 kernel 吃全宽） | 精度失败 | t_chunkdot/t_dot32：分块归约顺序 ≠ CANN 单趟全宽，MERE 超阈 |
| D=512 两半归约但不物化 workspace（边算边 softmax） | 精度失败 | 两半结果需先完整落 GM 再 softmax，否则跨块 rescale 引入超阈差 |
| fp32 档统一升精度路径 | case 40 max_error_cap 8/8192 违例 | verify3：fp32 输出舍入必须按 CANN 分档，一刀切 fp32 反而错 |
| `_fa_kernel` 两趟用于 D≤128 plain case | 慢于 online | bench2→bench3：两趟 K/V 扫两遍，online 单趟直接 +118% |

## 工件清单

- 实现：`0817/3/3_FusionAttention_triton_ascend_impl.py`（终态 700 行）及
  `verify/` 镜像；bench 证据 `verify/bench_run.log…bench_run8.log`（8 轮全 61/61）、
  `verify/verify_run*.log`（11 轮）、`verify/perf_result.json`（0.8068 终值）。
- 探针：`probe_f7.py`（语义）、`probe_precision.py`(精度路径)、`probe_twopass.py`
  （两趟）、`t_online_*.py`（online 系 5 个）、`t_2half/t_chunkdot/t_dot32/t_dc`
  （D=512 归约写法扫描）、`t_pse_ab.py`（pse stride 对照）、`t_3stage_block.py`
  （三段式 BLOCK 扫描）、`t_rtloop.py`（scf.for 非展开循环对照）。
- golden 锚：`0817/3/output/.baseline_anchor.json`
  （torch_sha256 `2b7f7b0b…`，冻结于 2026-08-17T08:16，未被改动）。
- 对照基线：`0810/3/attempt_from_old/`（`3_FusionAttention_final.py` + perf_result.json
  0.8914 + report.md；golden 锚 `c29eb586…`，与 0817 版仅输入生成参数两行不同，
  framework 侧 61 case 一致 ±4.5%，可比性成立）。
