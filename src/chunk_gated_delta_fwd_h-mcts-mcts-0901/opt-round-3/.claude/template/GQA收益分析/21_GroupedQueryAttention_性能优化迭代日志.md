# 21_GroupedQueryAttention 迭代日志

目标: 精度 50/50 全过前提下 speedup_vs_torch ≥ 3x。
工具链: triton 3.2.0 / triton_ascend 3.2.2 / CANN 9.1.0 / bishengir 1.2.0 (8796a8ac) / 910B2C(24 AIC, 48 AIV)。

⚠️ **v1~v4 跑在错的工具链上**(CANN 8.5.1 + bishengir 0.1.0)，其编译器相关结论已全部作废，
仅保留作为"证伪记录"。原因见 `_infra/AGENT_BRIEF.md` 第 1 节。

---

## 基线

`output/generated_code.py`，50/50 通过，**speedup = 0.0016**
（framework 0.1134 ms / impl 72.58 ms）。

瓶颈：`gqa_attention_kernel` 758389.7 us、`linear_kernel` 116.0 us —— attention 占 99.98%。
根因：基线 attention **一个 `tl.dot` 都没有**，是 `for d in range(d_head)` 的标量循环逐维累加
`scores += qd * kd`，PV 段用 `tl.sum(p[:,:,None] * v[None,:,:], axis=1)` 三维广播；
且 `BLOCK_SQ=1`、`BLOCK_SK=8`、两趟扫 K（先求 max 再求和，不是 online softmax）、
`num_cores` 硬编码 14（实测 num_aicore=24）。

---

## v1 — tl.dot 全面重写 ❌（编译失败，且工具链是错的）

改动：attention 改 `tl.dot` + online softmax；4 个 linear launch 合并为 2 个（QKV 三合一 + out）；
权重 `__init__` 预转置成 `[in,out]`；attention 直接按 `[B,S,H,dh]` 落盘省掉 permute；
权重 device/dtype 转换结果进 cache；`num_cores` 14 → 24。

结果：**verify 16/50**，34 个失败全是同一条编译崩溃
`hivm.hir.vsel op Unsupported op for finding the root alloc`。
通过的恰好是 S≤61 的 case（BLOCK=64），失败的全是 S≥67（BLOCK=128）。

## v2 — 删冗余 tl.where ❌（假设被证伪）

假设 A：崩溃由 `tl.where` 的数量/位置触发（vsel 即 tl.where，且 80 号算子历史日志记过同族崩溃）。
删掉 3 处冗余 where（q 侧、v 侧、p 侧）。

结果：**仍然 16/50，失败集合逐个相同** ⇒ **假设 A 证伪**，崩溃与 tl.where 无关。

## v3 — BLOCK_M/N 上限 128 → 64 ✅（假设被证实）

假设 B：崩溃由 BLOCK=128 触发。

结果：**44/50**。剩下 6 个失败恰好且仅仅是 dh∈{72,78,80} → BLOCK_DH=128 的那 6 个。
⇒ 假设 B 成立，但判据不是"某一维等于 128"。

## v4 — UB 预算联合定 tile ✅ 50/50（但性能数据无效）

编译探针 `scratchpad/probe_block.py` 直接扫 (BM,BN,BDH)：
`64x64x64` OK、`32x32x128` OK、`16x16x128` OK、`64x64x32` OK；
`128x64x64`、`64x128x64`、`64x64x128`、`128x128x128` 全崩。
⇒ 判据是**三维乘积 ≤ 2^18**（`32x32x128` 就是好的，所以不是"某一维 128"）。
按此规则定 tile 后 **verify 50/50**。

性能测量作废：benchmark 只测到 **13/50** 个 case 却仍算出 2.83x —— 别的会话在测量中途重装
triton，我方进程从第 14 个 case 起每次新编译都 `OSError('could not get source code')`。
已加两道闸门（必须测满 50 case、工具链指纹前后对照）。

## ⚠️ 工具链纠正（v1~v4 全部作废）

非交互 shell 读不到 `~/.bashrc`（第 6 行 `[ -z "$PS1" ] && return`），一直跑在容器层的
**CANN 8.5.1 + bishengir 0.1.0(e4e2ba98, 2026-02-06)** 上，而 `~/.bashrc` 第 134 行指定的是
**CANN 9.1.0 + bishengir 1.2.0(8796a8ac, 2026-07-29, llvm 19.1.7)**。
已固化 `_infra/toolchain.sh`，`envstamp.py` 逐轮校验版本。

## v5 — 在正确工具链上重扫 tile 预算 ✅ **1.4518x**

`scratchpad/probe_block2.py` 在 bishengir 1.2.0 上重扫：

| | fp16/bf16 | fp32 |
|---|---|---|
| `128x128x64` = 2^20 | OK | OK |
| `128x64x128` = 2^20 | OK | **UB overflow** |
| `256x64x64` = 2^20 | OK | — |
| `128x128x128` = 2^21 | UB overflow | — |
| `64x64x128` = 2^19 | OK | OK |

三点变化：(1) 预算 2^18 → **2^20**，整整 4 倍；(2) 失败模式从莫名其妙的
`vsel root alloc` 变成老实的 `UB overflow`；(3) fp32 下 BDH=128 比同乘积的 BDH=64 更吃 UB，
**不能只看乘积，要按 dtype 分档**。

⇒ v4 为绕开旧编译器把 tile 砍到 64/64（dh>64 时 32/32）的代价全部还回来。
**verify 50/50，benchmark 50/50，speedup = 1.4518**
（framework 0.1104 / impl 0.0760 ms；attention 70.68us、linear 128.76us）。

瓶颈画像：`linear_kernel` 6438us 占 **64.6%**，attention 3534us 占 35.4%；
`aic_mac_ratio` 仅 **0.0956**（Cube 基本没在算）、`aic_scalar_ratio` 0.7312、
`aiv_scalar_ratio` 0.6237、`cube_utilization` 92.7、`(aic+aiv)/Duration = 1.8537`（CV 串行）。
per-case：小 case 5~7x，大 case 只有 0.27~0.52x（最差 case45 B3S293D1344 fp32 = 0.274）。

## v6 — linear_kernel tile 按 dtype 分档 ✅ **2.2839x**

`work/sweep_linear.py` 在 v5 最差的 4 个 case 的 QKV 投影形状上扫 tile
（诊断性测量），相对原来拍脑袋的 64/128/64：

| 形状 | 最优 tile | 加速 |
|---|---|---|
| fp32 M=879 N=4032 K=1344 | 128/256/64 | **x3.56** |
| fp32 M=766 N=4416 K=1472 | 256/128/64 | **x3.46** |
| fp16 M=634 N=1536 K=1408 | 128/256/128 | **x5.15** |
| fp16 M=419 N=2048 K=1536 | 256/128/128 | **x4.87** |

低精度吃得下 BK=128，fp32 吃不下（`64/256/128`、`128/256/128`、`256/64/128`、`256/128/128`
全部 UB overflow）⇒ 按 dtype 分档：fp32 用 128/256/64，低精度用 128/256/128；
小形状按 `next_pow2` 收窄避免浪费。

**verify 50/50，benchmark 50/50，speedup = 2.2839**
（framework 0.1206 / impl 0.0528 ms）。
`linear_kernel` **128.76 → 35.88 us（3.6x）**，attention 71.47us 基本不变
⇒ 瓶颈易主，attention 成为主要开销。

## v7 — PV 段改单次低精度 dot ❌ **拒绝**（假设被证伪）

假设：参考实现 PV 段是 `softmax(fp32) @ v.float()`，v5/v6 为逐位同构用了 **fp32 操作数 dot**；
Cube 的 fp32 matmul 吞吐远低于低精度，`aic_mac_ratio 0.0956` 应有一部分记在这上面。

结果：**verify 18/50**，13 个 AccuracyError + 19 个 MLIRCompilationError。
- 精度：bf16 case8 `MERE=1.278e-02` > 阈值 `7.8125e-03`（超 1.6 倍）——
  与 80_GqaSparseFwd iter_0 的结论同族，单次低精度 dot 撑不住 fp32 golden。
- 编译：19 个 MLIR 失败，`probe_pv_cost.py` 后来查明原因是低精度路径多一份 p 的 cast 缓冲，
  在 128/128 tile 下直接 UB overflow。

**关键收获（比失败本身更重要）**：随后用 `work/probe_pv_cost.py` 单独量了三种 PV 写法的**速度**：

| shape | fp32 dot | 单次低精度 | hi/lo 双低精度 |
|---|---|---|---|
| c40 B3S897D512 fp16 tile128/128 | 480.5us | UB overflow | UB overflow |
| c38 B2S641D768 bf16 tile128/128 | 278.5us | UB overflow | UB overflow |
| c50 B3S463D1280 bf16 tile128/64 | 472.3us | 454.4us (**x1.04**) | UB overflow |
| c46 B2S317D1408 fp16 tile128/128 | 149.6us | UB overflow | UB overflow |

⇒ **fp32 PV dot 根本不是瓶颈**：唯一能编过的那档也只快 4%，还要赔上精度。
整条"换低精度 dot"路线（含 80 号那套 `p_hi/p_lo` 拆分）在 21 号算子上不成立。

## v8 — attention tile 低精度档 128x128 → 256x64（进行中）

依据 `work/sweep_attn.py`：低精度 dh≤64 档 256/64 比 128/128 快 3~8%；
fp32 档 128/128 已是最优（256/64 直接 UB overflow），dh=80 档 128/64 已是现状。
BLOCK_M 翻倍同时把 KV 扫描遍数减半（检查清单第 4 条）。

---

## 已排除方向（都有实测证据）

| 方向 | 结论 | 证据 |
|---|---|---|
| `tl.where` 数量/位置导致 v1 编译崩溃 | **证伪** | v2 删 3 处 where 后仍 16/50，失败集合逐个相同 |
| "tile 某一维不能是 128" | **证伪** | 旧编译器上 `32x32x128` 就是好的；真判据是乘积，且乘积上限本身随编译器版本变（2^18→2^20） |
| PV 段换低精度 dot（含 hi/lo 双 dot） | **证伪** | 精度超阈 1.6 倍；速度只 x1.04；多数 tile 下 UB overflow（`probe_pv_cost.py`） |
| 编译参数调优 | **无收益** | 逐个试 9 个：`multibuffer` x1.003~1.012、`enable_ubuf_saving` x1.006~1.014、`sync_solver` x1.000~1.016、`nolimit_mbuf` x0.999~1.011 —— 全在噪声内；`ops_reorder` 在 c40 上 **x0.759（有害）**；`enable_flatten=False` 编译失败；`enable_mixed_cv` x0.995（确认 950-only，本机 910B2C 无效）（`probe_compile_params.py`） |
| K 预转置消除每迭代的 `tl.trans` | **证伪** | fp32 档 207.8us vs 201.3us（**x0.969，更慢**）；低精度档 `[BDH,BN]` 布局直接 UB overflow（`probe_ktrans.py`） |
| attention tile 还有大空间 | **证伪** | 全档扫描后 128/128 只比各形状最优档慢 3~8%（`sweep_attn.py`） |

## IR 分析结论

`work/ir/out/gqa_attention_kernel_last_pass.mlir`（bishengir 1.2.0，case40 驱动）：

- `scf.for` 计 4 个，但实为**我写的 2 层循环 × Cube/Vector 两份**
  （`hivm.func_core_type` 66 处、`hivm.part_of_mix` 2 处、`get_sub_block_idx` 1 处，
  iter_args 是 `memref<64x64xf32, ub>` —— BLOCK_M=128 被 2 个 vector 子核对半分）。
  ⇒ **循环嵌套已是最小，没有 MQA 那种"合并双循环 / 压平 task-sq 循环"的空间。**
- 同步操作合计 **136 个**：`set_flag` 37 + `wait_flag` 37 + `pipe_barrier` 26 +
  `sync_block_set` 18 + `sync_block_wait` 18。
  与 MQA 的 117 个同族 —— 这正是画像里 `aic_scalar_ratio 0.7312` /
  `aiv_scalar_ratio 0.6237` 的真身（等待被计入标量管线），而不是"标量运算过多"。
- 这 136 个同步是 online softmax 每个 KV 迭代**固有的 Cube↔Vector 往返**
  （QK 在 Cube → scores 交给 Vector 做 softmax → p 交回 Cube 做 PV）。
  CV 重排（检查清单第 7 条 / Diff 7）已经应用；`(aic+aiv)/Duration = 1.85` 说明
  两条流水仍然基本串行。

⇒ **21 号 attention kernel 目前受限于 CV 串行，而非 tile / 精度 / 编译参数。**
在 UB 预算(2^20)约束下 BLOCK_M 无法再翻倍来摊薄这笔同步开销；
把 GQA 同组 G 个 q head 堆进一个 tile 也不能减少 KV 访存
（tile 行数被 UB 钉死，堆 head 就得减每 head 行数，任务数等比例上升，总访存不变）。

## 待办

- 权重未用 `register_buffer` 暴露，verify 的隐藏状态探针报 `HIDDEN_STATE_UNEXPOSED`
  （不影响精度判定，50/50 通过；但为可审计性应补上）。
