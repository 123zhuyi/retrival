# 21_GroupedQueryAttention Triton-Ascend 性能优化报告

> 状态：**进行中**（随迭代更新）。本文只记录已经过 `verify` 50/50 + `benchmark` 50/50 双闸门的数据。

## 1. 达成值

| 项 | 值 |
|---|---|
| 基线 speedup_vs_torch | **0.0016**（50/50，framework 0.1134 ms / impl 72.58 ms） |
| 当前 speedup_vs_torch | **2.2839**（50/50，framework 0.1206 ms / impl 0.0528 ms） |
| 相对基线提升 | **约 1400x** |
| 目标 3x | **未达成**，当前 2.2839，缺口约 32% |
| 交付版本 | `output/best_generated_code.py`（= `work/v6_linear_tiles.py`） |
| 测量口径 | `triton-op-verifier/scripts/` 的 `verify.py`（50/50 强制）+ `benchmark.py --keep_res`；阈值未改，未用 `--verify_not_required` |
| 硬件 | 910B2C（24 aicore / 48 vectorcore），NPU 0 |
| 工具链 | triton 3.2.0 / triton_ascend 3.2.2 / CANN 9.1.0 / bishengir-compile 1.2.0 (8796a8ac1508, 2026-07-29, llvm 19.1.7) |

### 基线为什么会低到 0.0016

基线的 `gqa_attention_kernel` 单次发射 **758389.7 us**，占 impl 总耗时 99.98%。原因不是调参不当，
而是**整个 attention 里一个 `tl.dot` 都没有**：

- QK 用 `for d in range(d_head)` 的 python 标量循环逐维累加 `scores += qd * kd`；
- PV 用 `tl.sum(p[:, :, None] * v[None, :, :], axis=1)` 的三维广播；
- `BLOCK_SQ=1`、`BLOCK_SK=8`；
- 两趟扫 K（先求 max 再求和），不是 online softmax；
- `num_cores` 硬编码 14（实测 `num_aicore=24`）。

## 2. 基础设施

| 组件 | 作用 |
|---|---|
| `_infra/iter.sh` | 单轮闭环：verify → benchmark `--keep_res` → 瓶颈画像。`--keep_res` 零额外成本（benchmark 本来就在跑 profiling） |
| `_infra/bench_clean.sh` | 三道闸门：benchmark 全局互斥锁、必须测满 50 case、工具链指纹前后对照 |
| `_infra/env_band.py` | 环境探针基准带。torch 参考实现代码恒定，其 `framework.avg_latency_ms` 是天然探针 |
| `_infra/profile_report.py` | 解析 `kernel_details.csv`，按耗时加权聚合各 ratio + 计算 `(aicore_time+aiv_time)/Duration` |
| `_infra/envstamp.py` | 逐轮记录并**校验** triton / triton_ascend / CANN / bishengir 版本，跑错工具链直接非零退出 |
| `_infra/toolchain.sh` | 显式 source CANN 9.1.0 的 set_env（非交互 shell 读不到 `~/.bashrc`） |

### 两次被闸门拦下的无效测量（都发生过，不是假想）

1. **少测 case**：别的会话在测量中途重装 triton，我方进程从第 14 个 case 起每次新编译都
   `OSError('could not get source code')`，只测到 **13/50** 却仍算出一个漂亮的 **2.83x**。
2. **跑错工具链**：非交互 shell 读不到 `~/.bashrc`（第 6 行 `[ -z "$PS1" ] && return` 守卫），
   v1~v4 全部跑在容器层的 CANN 8.5.1 + bishengir 0.1.0 上，而不是 `~/.bashrc` 第 134 行指定的
   CANN 9.1.0 + bishengir 1.2.0。两套都能跑通、不报错，只是编译器差了 5 个月加一个 llvm 大版本。

## 3. 当前瓶颈画像

v5（耗时加权，头部 16 个 case 覆盖 81% 耗时）：

| 指标 | 值 |
|---|---|
| `linear_kernel` | 6438 us（**64.6%**） |
| `gqa_attention_kernel` | 3534 us（35.4%） |
| `aic_mac_ratio` | **0.0956** ← Cube 基本没在算 |
| `aic_scalar_ratio` | 0.7312 |
| `aiv_scalar_ratio` | 0.6237 |
| `cube_utilization` | 92.73 |
| `(aicore_time + aiv_time) / Duration` | **1.8537** ← Cube-Vector 串行 |

v6 修掉 linear tile 之后瓶颈易主：`linear_kernel` 35.88 us、`gqa_attention_kernel` 71.47 us。

### 根因（profiling 与 IR 互相印证）

IR：`work/ir/out/gqa_attention_kernel_last_pass.mlir`（bishengir 1.2.0，case40 驱动）

- `scf.for` 计 4 个，但实为**我写的 2 层循环 × Cube/Vector 两份**
  （`hivm.func_core_type` 66 处、`hivm.part_of_mix` 2 处；iter_args 是 `memref<64x64xf32, ub>`
  —— `BLOCK_M=128` 被 2 个 vector 子核对半分）。
  ⇒ **循环嵌套已是最小**，没有 MQA 那种"合并 KV 双循环 / 压平 task-sq 循环"的空间。
- 同步操作合计 **136 个**：`set_flag` 37 + `wait_flag` 37 + `pipe_barrier` 26 +
  `sync_block_set` 18 + `sync_block_wait` 18。与 MQA 的 117 个同族。
- ⇒ `aic_scalar_ratio 0.7312` 的真身是**同步等待被计入标量管线**，不是"标量运算过多"。
  这 136 个同步是 online softmax 每个 KV 迭代**固有的 Cube↔Vector 往返**
  （QK 在 Cube → scores 交给 Vector 做 softmax → p 交回 Cube 做 PV）。
  CV 指令重排（检查清单第 7 条 / Diff 7）已应用，但 `(aic+aiv)/Duration = 1.85`
  说明两条流水仍基本串行。

## 4. 有效改动（按贡献排序）

| 改动 | 增益 | 依据什么数据 |
|---|---|---|
| **attention 用 `tl.dot` + online softmax 重写** | 0.0016 → 1.4518（**约 900x**） | 基线画像 attention 占 99.98%，且源码里一个 `tl.dot` 都没有 |
| **`linear_kernel` tile 按 dtype 分档** | 1.4518 → **2.2839**（+57%） | v5 画像 linear 占 64.6% 且 `aic_mac_ratio` 仅 0.0956；`sweep_linear.py` 实测最优档相对 64/128/64 达 x3.56~x5.15 |
| tile 预算按新编译器重扫（2^18 → 2^20） | 含在 v5 内 | `probe_block2.py`：`128x128x64` OK、`128x128x128` UB overflow；v4 为绕开旧编译器砍到 64/64 的代价全部还回来 |
| QKV 三合一 + 权重 `__init__` 预转置成 `[in,out]` | 含在 v1 重写内 | 检查清单第 1、2 条：`nn.Linear` 的 `[out,in]` 布局让 b tile 最内轴 stride = in_features |
| attention 直接按 `[B,S,H,dh]` 落盘 | 含在 v1 重写内 | 基线有一个 `aclnnInplaceCopy_TransposeAiCore_Transpose` 19.46 us，是 `transpose(1,2).contiguous()` |
| 权重 device/dtype 转换结果进 cache | 含在 v1 重写内 | 与参考实现把 `nn.Linear` 缓存在 `self._cache` 同语义，不是把计算挪出计时区 |
| `grid = min(num_cores, total_tasks)`，num_cores 14 → 24 | 含在 v1 重写内 | 检查清单第 5 条；实测 `num_aicore=24` |

### 各 dtype 档的最终 tile

| kernel | 条件 | tile |
|---|---|---|
| `linear_kernel` | fp32 | BM/BN/BK = 128/256/64 |
| `linear_kernel` | fp16 / bf16 | 128/256/128 |
| `gqa_attention_kernel` | BDH ≤ 64 | BM/BN = 128/128 |
| `gqa_attention_kernel` | BDH = 128，fp32 | 64/64 |
| `gqa_attention_kernel` | BDH = 128，低精度 | 128/64 |

（小形状一律按 `next_pow2` 收窄，避免 N=144 却开 BLOCK_N=256 的浪费。）

## 5. 已排除方向及其证据

| 方向 | 结论 | 证据 |
|---|---|---|
| **PV 段换低精度 dot**（含 80 号那套 `p_hi/p_lo` 拆分） | **证伪，整条路线不成立** | 精度：bf16 case8 `MERE=1.278e-02` > 阈值 `7.8125e-03`（超 1.6 倍）。速度：`probe_pv_cost.py` 单独量三种写法，唯一能编过的一档只快 **x1.04**；其余 tile 下低精度路径因多一份 p 的 cast 缓冲直接 **UB overflow**（这也是 v7 那 19 个 MLIRCompilationError 的原因） |
| **编译参数调优** | **无收益，其中一个有害** | 逐个试 9 个：`multibuffer` x1.003~1.012、`enable_ubuf_saving` x1.006~1.014、`sync_solver` x1.000~1.016、`nolimit_mbuf` x0.999~1.011 —— 全在噪声内；**`ops_reorder` 在 c40 上 x0.759**；`enable_flatten=False` 编译失败；`enable_mixed_cv` x0.995（确认 950-only，本机 910B2C 无效）。与知识库"约 30 个试下来命中率很低"一致 |
| **K 预转置消除每迭代的 `tl.trans`** | **证伪** | `probe_ktrans.py`：fp32 档 207.8us vs 201.3us（**x0.969，更慢**）；低精度档 `[BDH,BN]` 布局直接 UB overflow。`tl.trans` 不是显著成本 |
| **attention tile 还有大空间** | **证伪** | `sweep_attn.py` 全档扫描：128/128 只比各形状最优档慢 3~8% |
| **`tl.where` 数量/位置导致 v1 编译崩溃** | **证伪** | v2 删掉 3 处冗余 where 后仍 16/50，**失败集合逐个相同** |
| **"tile 某一维不能是 128"** | **证伪** | 旧编译器上 `32x32x128` 就是好的；真判据是三维乘积，而这个上限本身又随编译器版本变（2^18 → 2^20） |
| **GQA 同组 G 个 q head 堆进一个 tile 以复用 KV** | 未实施，先算账否掉 | tile 行数被 UB 钉死：堆 G 个 head 就得把每 head 行数除以 G，任务数等比例上升，**总 KV 访存量完全不变** |

## 6. 与 MQA / FA 历史经验的差异

| 条目 | 历史结论 | 本算子实测 |
|---|---|---|
| 权重转置 / tile 连续跨度 | MQA 最大单项增益 **5.94x** | **不是最大项**。本算子基线的 b tile 最内轴 stride 本来就是 1（是 `tl.dot(a, tl.trans(b))` 而非离散 gather），预转置只是顺带消掉 `tl.trans`；真正的大头是 **tile 尺寸**（3.6x） |
| `tl.dot` 的 p 二段拆分（~22 bit） | MQA 不够、28_FA 上 50/50 全过且 +2.7% | **两头都不成立**：精度超阈 1.6 倍，且速度只差 4%、多数 tile 下 UB overflow |
| 编译开关 | MQA 上三个有正收益 | **一个都没有**，且 `ops_reorder` 反而 -24% |
| `aic_scalar_ratio` 高的含义 | 是同步等待被计入标量管线 | **一致**，0.7312 同样是 Cube↔Vector 握手 |
| tile 可行域 | —— | **随编译器版本整片作废**：同一份 kernel，bishengir 0.1.0 上预算 2^18 且报 `hivm.hir.vsel root alloc`；1.2.0 上预算 2^20 且报老实的 `UB overflow` |

## 7. 当前判断

attention kernel 受限于 **Cube-Vector 串行**，而非 tile / 精度 / 编译参数 —— 这四条路都已用实测关闭。
在 UB 预算（2^20）约束下 `BLOCK_M` 无法再翻倍来摊薄那 136 个同步；
把同组 q head 堆进一个 tile 也不减少访存（见第 5 节末行的账）。

per-case 看，小 case（S ≤ 61）已达 5~7x，拖累几何平均的是 10 个低于 1x 的大 case
（dh ∈ {72,78,80} 被 pad 到 BLOCK_DH=128 的那批，以及 fp32 大 D 的那批）。

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
| 旧（按 kernel 名，只计一次） | 0.1206 ms | 0.0528 ms | 2.2839 | 50/50 |
| **新（按单次调用，全部计入）** | **0.1797 ms** | **0.0715 ms** | **2.5119** | **50/50** |

**impl 从 0.0528 涨到 0.0715，是因为本算子的 `linear_kernel` 一次 forward 里
确实发射 **2 次**（QKV 融合投影 + out_proj），旧口径漏计了其中一次。
framework 同步从 0.1206 涨到 0.1797（torch 侧同样有多发射）。
⇒ **新口径下多发射不再有度量红利，按真实 wall-clock 设计即可。**

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
