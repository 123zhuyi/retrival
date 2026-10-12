# 80_GqaSparseFwd 迭代日志（CANN 9.1.0 + bishengir 1.2.0）

工具链每轮由 `_infra/envstamp.py` 校验：triton 3.2.0 / triton_ascend 3.2.2 /
CANN 9.1.0 / bishengir-compile 1.2.0 (8796a8ac1508) / 910B2C 24 AIC + 48 AIV。

## 度量口径（先说清楚，否则数字会被误读）

`benchmark.py` 的 latency 是 **按 kernel Name 分组的单次发射平均耗时之和**。
本 impl 单 kernel 单发射 ⇒ impl latency == kernel 真实 duration。
torch golden 是三重 python 循环，发射同名 aten kernel 成百上千次却只被计一次 ⇒
**framework latency 几乎与 case 规模无关，恒在 0.085~0.130 ms**（实测 50 case 全部落在此带）。

⇒ `speedup_i = fw_i / impl_i ≈ 0.109 / impl_i`，**优化目标退化为"压 kernel 绝对时间"**。
达成 3x 需要 `geomean(impl) <= geomean(fw)/3 = 0.1090/3 = 0.0363 ms`。
（这是口径，不是真实加速比；真实 wall-clock 上 torch golden 慢得多。）

---

## iter base_tc —— 正确工具链上重测基线

- 实现：`work/v0_base.py`（= 交接时的 `output/optimized_code.py`，未改一行）
- 依据：历史 0.9438 / 8.5.1 上的 2.114 都作废，必须在 9.1.0 + 1.2.0 上重取起点。
- 精度：**50/50**
- speedup：**2.2904**（geomean impl = 0.0476 ms，geomean fw = 0.1090 ms）
- 判定：接受为基线。**换编译器本身带来 0.9438 → 2.2904，历史结论作废这条被证实。**

### 逐 case 画像（1-indexed，与 benchmark 输出一致）

| 区段 | case | impl(ms) | speedup |
|---|---|---|---|
| 小 TQ / D=64 | 1-5,10-15 | 0.007-0.023 | 4.2-10.0 |
| 中 | 16-35 | 0.038-0.062 | 1.7-3.0 |
| 大 TQ / D=128 / topk=16 | 36-50 | 0.080-0.201 | 0.57-1.44 |

拖累 geomean 的是 36-50 这一段（0-indexed 35-49），全部是 D=128、G=16 或 6、TQ 8~64。

### 反推的成本模型（关键）

用 `tasks = TQ*KH`、`iters = ceil(tasks/24) * rtopk`（24 AIC，每 program 串行扫 rtopk 页）：

| case(1-idx) | tasks | rtopk | 串行迭代数 | impl | 每迭代 |
|---|---|---|---|---|---|
| 5  | 4   | 1  | 1  | 0.0070 | — (固定开销 ~6-7us) |
| 6  | 4   | 16 | 16 | 0.0322 | ~1.7us |
| 39 | 96  | 16 | 64 | 0.2012 | ~3.1us |
| 42 | 256 | 5  | 55 | 0.1830 | ~3.4us |

每次迭代只搬 64KB（K+V 各 [128,128]bf16）、做 3 个 [16,128]x[128,128] 的 dot，
理论 <0.1us。实测 1.7-3.4us ⇒ **不是吞吐瓶颈，是每次循环迭代的固定开销**
（CV 跨核同步 + topk_idx→block_table 两跳串行标量 GM load + scf.for 的
sync_block_set/wait）。与 20_FlashAttentionV2 的实测结论一致：
「砍向量指令条数没有收益，只有减少迭代次数有效」。

⇒ 主攻方向定为 **减少循环迭代次数**（每次迭代 gather 多个 KV 页），
辅以 **删掉 p_hi/p_lo 双 dot**（少一次 dot + 一整条 bitcast RNE 向量链）。

---

## 探针阶段的一个大坑（必须先记下来，否则后面所有数字都会被误读）

我先用一个「参数化探针」`work/vx.py`（env 开关切 NPG / PV / KUM / OPT）做单变量 A/B，
它在 NPG=1、PV=split 时**理论上逐字符等价于 v0_base**，实测却慢 2.03x
（15 case SUM 2261.2 vs 1112.4）。所有基于 vx 的结论（"fp32 dot -15%"、
"NPG=2 只有 -4%"、"K 无掩码无收益"）**全部作废**。

用 4 个单差异变体从 v0_base 侧二分（cases 38/41/43/5，SUM us）：

| 变体 | 与 v0_base 的唯一差异 | SUM | 判定 |
|---|---|---|---|
| v0_base | — | 555.5 | 基准 |
| b1_vptr | `tl.load(k_ptrs + D)` 代替独立构造的 `v_ptrs` | 556.6 | 无影响 |
| b2_hoist | `tk_base/bt_base` 循环外提 | 555.6 | 无影响 |
| b4_cdiv | `range((rtopk+1-1)//1)` 代替 `range(rtopk)` | 557.9 | 无影响 |
| **b3_maskfirst** | **把 `pos/valid` 掩码计算挪到 `tl.dot` 之前** | **1212.0** | **2.18x 劣化** |

**结论：本 kernel 对「循环体内 Cube/Vector 指令发射顺序」极度敏感——把 3 行向量掩码
计算提到 QK dot 之前，整个 kernel 慢 2.18x。** Cube 必须等 AIV 算完 `valid` 才能发 dot，
一次迭代多一次 CV 往返。这是 BRIEF 检查清单第 7 条（CV 指令重排）的同一机理，
但在 MQA 上只值 +1.9%，在本 kernel 上值 **2.18x**。

⇒ 方法论修正：**任何"等价重写"的探针都必须先与原实现对齐一次**，否则会在退化的
工作点上做单变量实验，得到方向相反的结论。

---

## 已证伪的方向（都在干净基线 v0_base 上，cases 38/41/43/5，SUM us，越小越好）

| 变体 | 改动 | SUM | 相对 v0_base | 判定 |
|---|---|---|---|---|
| v0_base | — | 555.5 / 556.6 | 1.00 | 基线（两次复现一致） |
| I `v1_f32dot` | PV 段换 fp32 单 dot（BRIEF 4.2 推荐的"最有希望的一刀"） | 1002.7 | **1.80x 劣化** | ❌ |
| J `v3_gather NPG=1` | 每迭代 3-D gather 1 页 + reshape | 编译失败 | — | ❌ |
| K `v3_gather NPG=2` | 每迭代 gather 2 页, 迭代数减半 | 1086.3 | **1.95x 劣化** | ❌ |
| L `v3_gather NPG=4` | gather 4 页 | 编译失败（hivm-plan-memory） | — | ❌ |
| M1 `m1_scalarlim` | `valid = n_offs < (min(seq,q_abs+1) - blk*BS)` 单次向量比较 | 1538.4 | **2.77x 劣化** | ❌ |
| M2 `m2_kunmask` | K 也走无掩码 load（BLOCK_D==D） | 565.1 | 1.02（噪声内） | ⚪ 无收益 |
| **R1 `r1_pvfirst`** | **两个 PV dot 提到 `l_i`/`acc` 缩放之前发射（纯重排）** | **489.0** | **0.879 → -12.1%** | ✅ |

### 各条的证伪内容

- **fp32 PV dot 不可用**：BRIEF 4.2 说 `tl.dot` 支持 fp32 操作数且与 torch 逐位相同——
  数值上属实（MERE 从 2.2e-6 降到 5.5e-7），但**性能上是 1.8x 劣化**：
  910B Cube 的 MMAD 只有 fp16/bf16/int8 通路，fp32 dot 走模拟路径，且 `v.to(fp32)` 把
  V 的 UB 占用翻倍。⇒ `p_hi/p_lo` 双 dot 不能删，它是**性能**选择而不只是精度选择。
- **多页 gather 不可用**：目的是把 rtopk 次迭代压成 rtopk/NPG 次。实测 NPG=2 反而慢 1.95x，
  NPG=1/4 直接编译失败（`hivm-plan-memory` / `Unknown core type: llvm.func @malloc`）。
  3-D 页 gather + `tl.reshape` 会强制一次 UB staging，代价远超省下的迭代数。
  ⇒ 「减少迭代次数」这条从 20_FlashAttentionV2 带来的经验**在本 kernel 不成立**，
  因为这里减少迭代的唯一手段（页 gather）自身代价更高。
- **标量上界掩码不可用**：把 `valid` 从「两次向量比较」换成「一次向量比较 + 标量上界」
  慢 2.77x。标量上界依赖 `blk`（GM 标量 load）→ 标量链插进了向量流水。
  ⇒ 「减少向量指令条数」在本 kernel 上同样不是有效判据（与 20_FA 的结论一致）。
- **K 无掩码无收益**：历史日志记载这会触发 bishengir 非确定性崩溃；在 1.2.0 上它
  **既不崩也不快**（4 case 精度全过，时间 +1.6% 在噪声内）。K 侧 vsel staging 不是瓶颈。

## 唯一有效的一类改动：循环体内 Cube/Vector 指令发射顺序

| 变体 | 改动（都基于 v0_base） | SUM(38/41/43/5) | 相对 |
|---|---|---|---|
| v0_base | — | 556.6 | 1.000 |
| **R1 `r1_pvfirst`** | `pv = dot(p_hi,v)+dot(p_lo,v)` 提到 `l_i`/`acc` 缩放之前 | **489.0 / 486.7 / 488.6** | **0.876** |
| T1 | R1 + 删掉冗余 `p = where(valid,p,0)` | 486.8 | 1.000（对 R1） |
| T2 | R1 + `p_hi` 用截断(1 条 AND)取代 RNE 链(4-9 条) | 486.7 | 1.000 |
| T3/T4 | R1 + `tl.exp2` + log2e 折进 scale | 编译失败（本版 triton 无 `tl.exp2`） | — |
| U1 | R1 + `delta = exp(...)` 也挪到 dot 之后 | 488.8 | 1.000 |
| U2 | R1 + 截断 + delta 后移 | 486.6 | 1.000 |
| U3 | U2 + `tl.maximum(propagate_nan=ALL)` | 487.0 | 1.000 |

**所有"减少向量指令条数"的改动一律 0 收益**（截断 vs 9 条 RNE 链、删一次 where、
`propagate_nan`、`delta` 后移，全部落在 ±0.5% 内）。**只有改变 Cube 指令相对于
Vector 指令的发射位置有收益**（R1 -12.4%，b3 反向 +118%）。

⇒ 本 kernel 的瓶颈画像结论：**Cube 与 Vector 的串行等待**，
AIV 的算术量、AIC 的 MAC 量都不是瓶颈（`aic_mac_ratio` 仅 6.6%）。
