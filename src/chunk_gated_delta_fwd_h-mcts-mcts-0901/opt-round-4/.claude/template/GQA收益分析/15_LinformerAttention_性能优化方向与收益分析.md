# 15_LinformerAttention 性能优化方向与收益分析

> 面向"下一个做 Linformer / 低秩投影类 attention 算子的人"。每条标注**实测收益**与**适用条件**。
> 实测环境：Ascend 910B3（Ascend910_9372，**20 AIC**，注意与 GQA 家族的 910B2C/24 AIC 不同 SKU）、
> triton 3.2.0 / triton_ascend 3.2.2 / CANN 9.1.0。与 `21/80/81/99` 号经验多处冲突，冲突处以本文实测为准（§5）。
> ⚠️ 数据来源分级：R1–R10 的数字盘上 `triton_ascend_output/…/perf_result.json` 可验证；
> v2–v16 手工线的 `.py`/`perf_result.json` **已被 2026-08-14 09:17 的清理误删**，数字仅存
> 会话 transcript 与 `.pyc`（v15b 的 BLOCK_S=256 已用字节码反汇编核实），下文逐条标注。

## 0. 算子形态（先算账）

```
x [B,S,D] --w_q/w_k/w_v--> q[B,H,S,hd]  k,v[B,H,hd,S]      # 3 个 D×D 投影
k --w_kr(seq→P)--> kr[B,H,P,hd]  (v 同)                    # 2 个 S×P 序列降维
scores = q.float() @ kr.float().T                            # golden 在 fp32 域
out = (softmax(scores/√hd) @ vr.float()).to(dtype)
out --w_o--> [B,S,D]                                         # 1 个 D×D 投影，共 6 个 GEMM
```

- 50 个 case：B ∈ [1,4]，S ∈ [3,897]，D ∈ [48,1536]（44 档），H ∈ [2,24]，
  P(=projection_dim) ∈ [2,53]，hd ∈ {20,…,80}；fp16/bf16/fp32 各 17/17/16 个。
- **P/S 中位数 0.15**：attention 矩阵是"短胖"形（N=P 极小），这与 FA/GQA 家族的
  "KV 长循环"形态完全不同——**FA 那套 KV 循环握手经验在这里基本不适用**。
- **三个决定方向的事实**：
  1. **瓶颈在 6 个投影 GEMM + 布局拷贝，不在 attention 数学**。S 小 D 大 ⇒
     投影 FLOPs `2·BS·D²·(4次)` 远大于 attention `2·B·H·S·P·hd·2`（P≪S）。
  2. **torch 参考侧约一半算子时间在纯搬运**（Transpose 77.0 + 46.4 + 5.1 µs）——
     参考实现里有 4 处 `permute().contiguous()`/`transpose().contiguous()`，这是 impl 侧最大的可赢面。
  3. **P ≤ 53 且单 tile 放得下** ⇒ softmax 是**单 pass**（不需要 online softmax 跨迭代），
     attention kernel 的 CV 握手成本天然低于 FA。

## 1. 收益排序（本算子实测，两条线合并）

| # | 方向 | 实测收益 | 适用条件 |
|---|---|---|---|
| 1 | **attention 从"逐 token 标量点积"改为 `tl.dot` 分 tile** | 0.089 → 0.96（**约 11x**，R2） | 基线 attention 里没有 `tl.dot`、`grid` 出现 `total=B*H*S` 时 |
| 2 | **重生成架构**：通用投影 GEMM kernel + 融合 attention kernel（R5 换草图） | 1.24 → 1.37（R5 Phase3 直接跳） | 每轮 worker 携带历史经验重写，比在烂架构上调参值钱 |
| 3 | **消除 attn 输出侧 `.contiguous()`：kernel 按 `[B,S,H,hd]` 落盘 + `reshape(M,D)` 免费** | 1.494 → **1.7393（+16%，v4）** | 参考实现有 `transpose(1,2).contiguous()` 时，kernel 带 stride 直写目标布局 |
| 4 | **消除 k/v 侧 2 次转置拷贝：专用 strided-reduce kernel 直读 `k2d[B*S, H*hd]`** | 1.7393 → **1.9921（+14.5%，v6）** | 有"先物化投影再做序列降维"的两段式时，把 permute 折进 reduce kernel 的寻址 |
| 5 | **kv_reduce BLOCK_S 上限 64 → 256** | 1.9921 → **2.0119–2.0315（破 2.0×，v15b）**；kv_reduce kernel −22.5%（6.48→5.02） | 大 S case；UB 预算内加深 K 维（见 §2.5 账） |
| 6 | **attention 任务分区改 (b,H)-major 连续划分 + Kr/Vr 外提** | 1.4061 → 1.4784（+5.1%，R8）；R9 再 +0.1% | NUM_S>1（S 超过单 tile）的 case；Kr/Vr 每 S-tile 重载时 |
| 7 | **GEMM tile 按 dtype 分档 + Swizzle2D GROUP_M=4** | 1.3673 → 1.409（+3.1%，R5） | 低精度 BK=128 / fp32 BK=64；与 21 号结论同族 |
| 8 | **fp32 走低秩结合律重排 k/v 路径**（`(x@W_k^T)@W_kr^T → W_k@(x^T@W_kr^T)`） | v3 = 1.494（贡献并入 dtype 分派） | **仅 fp32**；低精度会因中间舍入差 1 ULP 超阈（见 §4 证伪） |
| 9 | **attention block 尺寸按形状自适应（next_pow2 + cap）** | 0.9614 → 1.0498（+9%，R2） | hd/P/S 分布广（本算子 hd 20~80、P 2~53）时 |

## 2. 有增益方向详解

### 2.1 布局拷贝是本算子的主战场（#3/#4/#5 是同一条线）

参考实现的 4 处搬运与对应消除法：

```python
# ❌ 参考路径（4 次搬运 + 2 个大 GEMM 物化）
k = k_proj(x).view(b,s,H,hd).permute(0,2,3,1).contiguous()   # 搬运 1：[B,H,hd,S]
v = ...                                                       # 搬运 2
kr = key_reduce(k.view(B*H*hd, S)).transpose(-2,-1).contiguous()  # 搬运 3
vr = ...                                                       # 搬运 4
attn_out = attn_out.transpose(1,2).contiguous()               # 搬运 5（输出侧）
```

- **输出侧（v4）**：attention kernel 的 o_ptr 直接按 `[B,S,H,hd]` 的 stride 落盘
  （`stride(0), stride(2), stride(1), stride(3)`），host 侧 `attn.reshape(M, d)` 是免费的（连续内存重解释）。
- **k/v 侧（v6）**：专用 `_kv_reduce_strided_kernel` 直读投影输出 `k2d [B*S, H*hd]`，
  `kr[b,h,p,hd] = Σ_s k2d[b,s,h,hd]·w_kr[p,s]`，s 方向按 stride=D 跨步，
  一个 program 处理一个 (b,h)、M=HD tile —— **把 permute 折进寻址而不是先搬再做**。
- **v15b**：该 kernel 的 BLOCK_S 上限 64→256（当时自 v6 起冻结了 5 个版本没动）。
  UB 账：`acc[64,128]fp32=32KB + w[64,256]fp16=32KB + k[256,128]fp16=64KB = 128KB < 192KB`。
  收益来自两处：K 迭代次数减半/四分之一 + 每次 dot 的归约深度加大（MAC 利用率）。
  求和重组误差 ~1e-7（tile 尺寸变化，不是代数重排）仍在 MERE 闸门内，50/50 逐位过。

> **可迁移判据**：forward 里每出现一次 `.contiguous()`，先问"这次搬运能不能用 stride 消掉"
> （与 21 号同结论）；**消不掉时，第二选择是写专用 kernel 把转置折进寻址**（本算子 v6 路线），
> 而不是让现有 GEMM kernel 走 strided load（那条 R7 试过，灾难，见 §4）。

### 2.2 attention 任务分区（R8/R9，与 99 号家族同族）

坏味道：`for tile in range(pid, NUM_TILES, num_cores)` 的交错分区，让同一个 program 的
连续迭代落在**不同 (b,H)** 块上 ⇒ Kr/Vr（每 (b,H) 一份）在 HBM 里被反复重载，
流量 O(B·H·NUM_S)。

R8 改为连续分区 `[pid·TPP, (pid+1)·TPP)` + Kr/Vr 外提到 S-tile 循环外 ⇒ 流量 O(B·H)，
大 S case 收益最大（case41 0.522→1.001、case46 1.202→1.655），sub-1x case 6→2。
R9 再做 NUM_S 自适应双 kernel 分派：NUM_S>1 走 hoist、NUM_S==1 走原 strided
（小 NUM_TILES 下 strided 保 L2/占用率），attention 占比 35.5%→26.9%。

⚠️ **同一招用在投影 GEMM 上无效**（R8 opt_iter_1）：K1 的 Swizzle2D + 交错持久化
本来就是 L2 调优过的形态，连续分区 −0.15%（噪声）。

### 2.3 dtype 契约（本算子两条硬规则）

1. **fp32 `tl.dot` 在 910B3 上逐位精确**（`allow_tf32=False`，实测 maxerr=0）——
   template `attention.md` L1.1 的"fp32 tl.dot 禁令"是老芯片写的，**本代芯片不适用**。
   attention 的 `q.float()@kr.float()` 直接 fp32 dot 逐位同构，不需要 80/81 号的 `p_hi/p_lo` 拆分。
2. **低秩重排只在 fp32 安全**：`W_k@(x^T@W_kr^T)` 数学等价，但低精度下中间舍入比参考
   （参考先把 `k_proj(x)` 舍入到低精度再降维）差 ~1 ULP ⇒ 无条件套用实测 **34/50 失败**
   （bf16 阈 7.8e-3 / fp16 阈 9.8e-4）。生产版按 `dtype==float32` 分派两条路径。

### 2.4 工具链/环境

- 910B3 `num_aicore=20`，`_get_num_cores()` 运行期读取（R5 #3 命中）；硬编码 14/24 都不对。
- benchmark 口径：`speedup_vs_torch` 是**逐 case 几何平均**；warmup=5 / repeats=50；
  verify 50/50 是硬闸门（阈值未改、未用 `--verify_not_required`）。

## 3. 先算账：为什么天花板在投影 GEMM

探针实测（R10 #13）：`[838,1536]@[1536,1536]` fp16，triton `_mm` 最优配置
（num_warps=4/num_stages=3，扫过 {4,8}×{2,3,4}）= 0.059 ms，CANN `aclnnMatmul` = 0.027 ms
—— **triton 通用 GEMM 比手调库慢 2.2x，这是编译器/库级 gap，调参闭合不了**。
而 AST 退化预检禁止 forward 里用 `torch.matmul`（`torch_npu.*` 也不行）⇒ 投影段被钉死。

R10 当时据此推算"geomean 天花板 ~1.4–1.5x"——**这个推算被 v4/v6/v15b 证伪**：
它假设 4 处布局拷贝必须保留。把拷贝消掉（+16%、+14.5%、+1% 三刀）后，生产版在同一
framework 基线下打到 2.03×。**教训：天花板估算必须把"可消除的非计算开销"先归零再算。**

生产版最终形态的 impl 时间构成：投影 GEMM ~38% + attention ~48%（均 cube-bound）、
kv_reduce ~14%——三条都被 cube 天花板盖住后，2.2× 目标的剩余缺口在大 D fp32 case
（对 aclnnMatmul 无还手之力，结构性 0.4–0.9×）。

## 4. 已证伪方向（比 §2 更值钱）

| 方向 | 结果 | 证据 |
|---|---|---|
| **q/k/v 投影权重 host-concat 成单 GEMM** | **0.973（−29%），且 9/50 精度漂移** | R5 opt_iter_0：concat 改变 RNG/精度调度 + 大矩阵 tile 并行度下降。R6 的 launch-fusion 变体（合并发射、平面布局 [3,M,D]）0.9387（−32.5%），50 case 全变慢 |
| **launch-fused 投影 kernel 内按 tile 算 g 偏移** | 同上，R6 根因记录 | 每迭代变更权重/输出基址指针 **摧毁编译器对 K 归约循环的软件流水**；省 2 次 launch 抵不过每 tile 变慢 |
| **GEMM 内 strided-gather load 消除 k/v 转置拷贝** | **case49 慢 11.78x、case37 慢 1.86x** | R7 #31：A-load 沿 s 以 stride=H·hd（1280–1440 元素）跨步，退化为不可合并 gather；被替代的高度优化转置拷贝反而便宜 |
| **kv_reduce 融进 attention kernel（每 (b,h) 做 M=hd 的小 GEMM）** | 0.8882（attention 占 88% 成瓶颈） | R10 opt_iter_0：M=hd 的 dot 是 latency-bound，小 P 下 N 侧浪费；拆出来按 `M=B·H·hd` 大 M 单独做 GEMM 才能喂饱 cube |
| **W^T 以 swapped-stride 在 kernel 内加载**（R10 基线形态） | **0.0525（27/50 case <0.1x）** | R10 iter_0；改 host 预转置 `[in,out]` + 干净 a@b 后单点 **26x**（→1.3708）。⚠️ 与 R5–R9 同写法测得 1.37–1.48x 矛盾，说明该退化对 kernel 其余结构敏感，不能孤立的当作普适结论；但"预转置+干净 a@b"是稳健优选 |
| **qkv 三输出/双输出融合 kernel** | v8=1.388（−30%）、v12=1.5039（−24%） | 融合迫使 BM/BN 或 BK 砍半以塞进多份 fp32 累加器 ⇒ cube MAC 损失 ~25%，**没有任何融合能赔回**。本芯片上"多输出 GEMM 融合是死路" |
| **attention dot 换 fp16 操作数**（fp32 累加） | v13=1.8946（−5%），精度 50/50 过 | 910B3 fp16 dot 确实走 fp32 累加（逐位过），但 **cube 在小 tile（BP≤64, HD≤128）上是 latency-bound 不是吞吐 bound**，fp16 的 2× MAC 率兑现不了 |
| **收紧 BK=128 的生效守卫（只对大 GEMM）** | 1.409 → 1.3541 | R5 opt_iter_2：小 GEMM 也吃 BK=128 的 MAC 收益 |
| **GROUP_M 扫描 8 / 2** | v9=1.9836、v10=1.9921（≈v6，噪声内） | GROUP_M=4 即饱和 |
| **num_stages 流水参数** | v14 无变化 | 投影/attention 均 cube-bound，流水参数不动瓶颈 |
| **手写 host 侧 per-op 计时器定位瓶颈** | **结论全错**（误判 launch-bound） | Python lambda + `time.time` 单次调用把每个 op 都摊成 ~0.23–0.26ms 常数；benchmark 实际测的是 profiler 的 kernel `Duration(µs)`。**定位瓶颈必须用 kernel_details.csv，不要用 wall-clock 包 python 调用** |

## 5. 与 GQA 家族经验的冲突对照

| 条目 | GQA 家族结论 | 15 号实测 | 冲突原因 |
|---|---|---|---|
| attention 主成本 | 80/81/99：KV 循环 CV 握手主导（4–6µs/页） | **不适用**：P≤53 单 tile softmax 单 pass，无跨迭代握手 | 形态差异：P/S 中位 0.15，"短胖" attention |
| `p_hi/p_lo` 双 dot | 80 号：精度必需且是性能正选择 | **不需要**：910B3 fp32 dot 逐位精确 | 芯片代际：L1.1 禁令是老芯片；且本算子 dot 的 N=P 极小 |
| fp16 dot 提速 | 常规预期 2× MAC | **更慢 5%**（v13） | 小 tile latency-bound；80 号"dot 个数是成本单位"在握手主导形态才成立 |
| GEMM tile 分档 | 21 号：**最大单项** +57% | **仅 +3.1%**（R5 #2） | 基线差距不同：21 号基线 tile 拍脑袋 64/128/64，15 号 R5 基线已接近优 |
| 多输出 GEMM 融合 | —— | **两败俱伤**（v8/v12，−24~−30%） | UB 多份 fp32 累加器逼 tile 减半，cube 损失主导 |
| 消 `.contiguous()` | 21 号：stride 消掉（1.8x 分水岭） | **一致且加强**：v4/v6 两级消除 +16%/+14.5%；但"stride 直读"失败（R7），**要靠专用 kernel 折进寻址** | 投影输出后还要做序列降维，转置与归约可合并 |
| 历史结论会作废 | 80 号：换编译器整片翻转 | **一致**：L1.1 fp32 dot 禁令在本代失效 | —— |

---

## ★ 测量口径变更：同一份 v15b 代码 2.03× → 2.37×

2026-08-14 用**更新后的 benchmark.py** 复测 v15b（kernel 一字未改、50/50 逐位过）得
**逐 case 几何平均 2.3747×**（32/50 case >2.0×）。原因：新 benchmark 测得的 torch_npu
framework 基线慢约 3 倍（~0.23ms vs 旧 ~0.078ms），系统性抬高 speedup。
已排除频率因素：warmup=50 的运行仅 2 次频率漂移事件（NPU 稳定 1800MHz），framework 时延仍 ~0.23ms。

⇒ **报数必须绑定口径**：历史生产记录 = **2.0119–2.0315×**（旧 benchmark，0.078ms 基线）；
当前 benchmark 口径 = **2.37×**。两者描述的是同一份代码。
另：本算子全程无频率锁可用（uid 1008 非root，`dsmi_set_device_info` rc=46），
用"高 warmup + 漂移事件计数"替代验证稳定性。

## ★★ 2.0× 破关的复盘（2026-08-13）

从 1.48×（R9，编排器线终点）到 2.03×（v15b，生产）的 5 刀全部是**非计算开销消除**：

| 刀 | 改动 | 增益 |
|---|---|---|
| 1 | fp32 低秩重排 k/v 路径（dtype 分派） | 并入 v3（1.494） |
| 2 | attn 输出 `[B,S,H,hd]` 直写 + reshape（v4） | +16% → 1.7393 |
| 3 | k/v 转置折进 strided-reduce kernel（v6） | +14.5% → 1.9921 |
| 4 | kv_reduce BLOCK_S 64→256（v15b） | +1% → **2.0119–2.0315，破关** |

而编排器线 R6/R7/R10 三次尝试消除同样的拷贝全部失败（融合毁流水 / strided-gather 灾难 /
融合毁 cube）——**同一目标，"折进专用 kernel 寻址"这条唯一通路是被手工线试出来的**，
两次失败的自动尝试（#20/#31）都错在让现有 GEMM kernel 直接吃非连续布局。

### ⚠️ 工件事故（写文档的人必须知道）

2026-08-14 09:17 的清理**误删了 v3–v16 的全部盘上工件**（`.py`、`perf_result.json`、
`report.md`），生产文件 `15_LinformerAttention.py` 也一并消失。当前状态：
- v15b 代码仅存 `v15b_work/verify/__pycache__/*.pyc`（BLOCK_S=256 已经字节码反汇编核实）；
- v3–v16 的速度数字仅存会话 transcript；
- 盘上 `triton_ascend_output` 的 JSON（最高 1.48×）是**更早的编排器线**，**不代表生产成绩**。
⇒ 教训：**突破性版本必须立即备份 .py 源文件与 perf JSON 到独立目录**；
扫描磁盘下结论前先确认没有"空 v*_work 目录 + 幸存 .pyc"的指纹。
