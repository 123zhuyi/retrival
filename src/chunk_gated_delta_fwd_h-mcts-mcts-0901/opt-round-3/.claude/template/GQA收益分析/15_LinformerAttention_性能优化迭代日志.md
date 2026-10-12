# 15_LinformerAttention 迭代日志

目标: 精度 50/50 全过前提下 speedup_vs_torch ≥ 2.2（2.0 为硬关）。
环境: Ascend 910B3（Ascend910_9372，20 AIC / 40 AIV）/ triton 3.2.0 / triton_ascend 3.2.2 /
CANN 9.1.0 / bishengir-compile（pip 包自带）。verify+benchmark 阈值未改。

两条线：**R1–R10** 由编排器（fusion-search worker + MCTS 策略，GLM 网关 `glm-5.2`）驱动，
工件在 `triton_ascend_output/15_LinformerAttention/opt-round-N/`（盘上可验证）；
**v2–v16** 是主线（人工迭代，verify harness 在 `v*_work/verify/`）。
⚠️ 2026-08-14 09:17 清理误删 v3–v16 的 .py/perf JSON/report，数字仅存 transcript 与 .pyc；
下表 v 系列数字来自该 transcript（含多次复跑）。R 系列数字全部来自盘上 perf_result.json。

## 时间线总览

```
R1      R2        R3      R4       R5         R6     R7     R8        R9       R10     （编排器线）
0.089 → 1.0498 → 0.5865 → 1.2388 → 1.3673* → 1.392 → 1.362 → 1.4061 → 1.4784 → 1.3463**
        (*R5 Phase3 基线 1.3673，opt 后 1.409；**R9 基线因 profiler flake 49/50)

v2     v3        v4         v6         v15      v15b       （人工线，基于 R9 代码）
0.77 → 1.494 → 1.7393 → 1.9921 → 1.9992 → 2.0119–2.0315 ★破2.0 → (2026-08-14 复测 2.3747，口径变更)
```

---

## 编排器线（R1–R10）

### 基线（Phase 3 初版，R1 iter_0）

**speedup = 0.089**（impl 0.8512 ms / fw 0.0757 ms，50/50）。
根因：`linformer_attn_kernel` 的任务粒度是**单 token**（`total = B*H*S`），
P 维逐行做 `tl.sum(q * k_row)` 标量点积 + online softmax 的 m/l/acc 全是 `[1,]` 标量张量——
**一个 `tl.dot` 都没有**，与 21 号基线同病。6 个投影 GEMM 用 64/64/64 tile。

### R1 — head-tile attention + 动态 tile ❌

attention 改按 head 分块 + tile 按形状选：0.0694 / 0.0707（比基线还差）。
Phase 3 代码保留。

### R2 — attention tile 化（首个破 1x）✅ 1.0498

- opt_iter_0：attention 改 `[BLOCK_S, BLOCK_HD]` tile + `tl.dot` 做 Q@K^T 与 weights@V
  （固定 BLOCK_S=32 / BLOCK_HD=128 / BLOCK_PROJ=64）⇒ **0.9614**（10.8x）。
- opt_iter_1：`_attn_block_sizes(s,proj,hd)` 按形状 next_pow2+cap 选 block ⇒ **1.0498**。
- opt_iter_2：加 `_matmul_block_sizes`（GEMM tile 按 M/N/K 分档）⇒ 1.0434（略回，采纳 iter_1）。
- 注意：R2 的 attention tile 化 kernel 里 Q/Kr load 后立即 dot、softmax 单 pass——
  这套结构一直用到最终生产版。

### R3 — 换一种 attention 写法 ❌ 0.5865（回退）

opt_iter_1 把 attention 改成"BLOCK_S 行 Q + P 维逐行向量归约 + online softmax"
（试图避开 Ascend dot 限制）：0.5865。opt_iter_3 变体 0.5308。
**证伪**：在 P ≤ 53 的形态上，online softmax 的标量 m/l 往返比单 tile 单 pass softmax 贵得多。
P 小到单 tile 放得下时，**不要用 online softmax**——直接 m=max → exp → sum → div。

### R4 — 大 tile + 基因库路线 ✅ 1.2388

attention BLOCK_S 提到 128（大形状）+ G003/G005 基因；Phase3 基线本来自带 stacked QKV
投影（`w_qkv` host-concat，R2 已有）⇒ 1.0387 → 1.2388。
（注：R4 的 attempts 用另一套 30 点清单 T1–T30，命中 T1/T2/T3/T5/T6/T14/T15/T16/T17/T26/T27/T28/T29。）

### R5 — 架构重写 + 30 点清单全扫 ✅ 1.3673（Phase3）/ 1.409（opt）

Phase 3 重写为两 kernel 架构：`_proj_matmul_kernel`（通用 GEMM，持久化 grid，
W 按 [N,K] 转置 tile 加载）+ `_linformer_attn_kernel`（融合 scores/softmax/output）。
数字：iter_0 = **1.3673**。
- opt_iter_0：**#20 host-concat q/k/v 权重** ⇒ 0.973（−29%）**且 perf 出现 9/50 精度漂移**
  （split+reshape 非连续拷贝引入数值差异 + 大矩阵 tile 并行度下降）。**反模式，回退。**
- opt_iter_1：#3 `_get_num_cores()` 运行期读 num_aicore=20 + #2 Swizzle2D GROUP_M=4 +
  #28 W 转置 tile + BK dtype 自适应（lowp 128 / fp32 64）⇒ **1.409** ✅ 采纳。
- opt_iter_2：#13 收紧 BK=128 守卫（仅大 GEMM）⇒ 1.3541 回归。
  **证伪**：小 GEMM 也吃 BK=128 的 MAC 收益。

### R6 — launch-fused q/k/v ❌ 0.9387（回退）

3 个投影 GEMM 合并成 1 个持久化 kernel（tile 空间 G=3×NUM_M×NUM_N，输出平面布局 [3,M,D]，
堆叠权重预缓存）。50/50 逐位过（tile 算术相同），但**50 case 全部变慢**（−0.127 ~ −0.820，
中/大 D 最差）。
根因（记录得很细，值得抄）：kernel 每 tile 用 `g = tile // TILES_PER_GEMM` 算
数据依赖的权重/输出基址（`w_g_base`/`y_g_base` 每迭代变）⇒ **摧毁 K 归约循环的软件流水**
（单 GEMM 时权重基址全 kernel 固定，流水 + Swizzle2D L2 复用才成立）。
省 2 次 launch 抵不过每 tile 变慢。**列为 AVOID 模式（operator_experience #31）。**

### R7 — strided-gather 消 k/v 转置拷贝 ❌（probe 阶段即否）

目标：消掉 2 处 `permute().contiguous()`（msprof 测得占 impl ~15.7%）。做法：投影输出
`k2d/v2d [b*S, H*HD]` 直接喂 4-D 跨步 reduce matmul（固定输出行沿 s 以 stride=H·hd 跨步收缩）。
稳态 probe（30 iters）：case0 1.113x 慢、case37 1.858x 慢、**case49 11.78x 慢**（H·hd=1280/1440），
仅最小 hd 微赢。根因：大 stride 下 A-load 退化为不可合并 gather + per-row divmod 基址串行化。
**被替代的高度优化的转置拷贝比 strided-gather 便宜** —— 与后来 v6 的"专用 kernel 折进寻址"
成功路线对比：**差别在于 v6 只跨 s 一维且以 (b,h) 为程序粒度、每行 HD 连续，
R7 试图让通用 GEMM kernel 直接吃 4-D 散射布局**。

### R8 — attention (b,H)-major 连续分区 + Kr/Vr 外提 ✅ 1.4784

只动 attention kernel 的任务分区（投影/权重/tile 公式不动）：
交错 `range(pid, NUM, num_cores)` → 连续 `[pid·TPP, (pid+1)·TPP)`，Kr/Vr 提到 S-tile 循环外，
每 (b,H) 块加载一次复用 NUM_S 次（HBM 流量 O(B·H·NUM_S)→O(B·H)）。
逐 tile 算术不变 ⇒ 50/50 逐位过。1.4061 → **1.4784**（+5.14%），
大 S 目标 case：case41 0.522→1.001、case46 1.202→1.655，sub-1x 6→2；
代价 ~24 个小/中 case 微退（占用率/L2 效应）。
- opt_iter_1：同思路用到投影 GEMM（连续分区）⇒ 1.4762（−0.15%，噪声）。
  **证伪**：Swizzle2D + 交错持久化对 GEMM 已经是 L2 调优形态，§6 连续分区不迁移。

### R9 — NUM_S 自适应双 kernel ✅ 1.4800（终版：1.48x 备份文件）

NUM_S>1 走 R8 hoist kernel；NUM_S==1 走 R5 strided kernel（小 NUM_TILES 下 strided 保 L2/占用率），
两 kernel 内 S-tile 主体逐字相同（逐位一致）。基线 1.3463（49/50，case24 profiler flake）
→ **1.4800**（+9.93% vs 基线；36 wins / 9 噪声级微退）。
attention 占比 35.5% → 26.9%。30 点清单全部评估完毕，退出 Phase 4。
（R9 代码即后续 v 系列的起点，备份 `15_LinformerAttention_round9_1.48x.py`。）

### R10 — 从头再走一遍验证经验（含一次大翻车）✅ 1.3708

Phase 3 iter_0 复刻"round-8/9 获胜设计"时引入 **W^T swapped-stride 加载** ⇒
**0.0525**（27/50 case <0.1x，cube mmad 降级）。
- opt_iter_0：把 key/value_reduce 融进 attention（每 (b,h) 做 M=hd 小 GEMM）⇒ 0.8882，
  attention 占 88% 成瓶颈。
- opt_iter_1：**host 预转置权重 `[in,out]` + 干净 a@b + reduce 拆成大 M 单 GEMM** ⇒ **1.3708**
  （vs 本轮基线 +26x；绝对值略低于 R9 的 1.48，因架构细节差异）。
  两个结论进了经验库：(a) reduce 类 GEMM 必须以大 M 喂 cube；(b) W 预转置优于 kernel 内 swapped-stride。
- #13 探针：triton `_mm` vs CANN aclnnMatmul 在 [838,1536]@[1536,1536] fp16 上
  **0.059 vs 0.027 ms（2.2x gap）**，num_warps/num_stages 扫描闭合不了 ⇒ 库级天花板。

---

## 人工线（v2–v16，基于 R9 1.48x 代码）

| 版本 | 改动 | speedup | 判定 |
|---|---|---|---|
| v2 | （早期失败变体） | 0.77 | ❌ |
| v3 | **低秩结合律重排 k/v 路径**：`(x@W_k^T)@W_kr^T → W_k@(x^T@W_kr^T)`，中间量从 [M,d] 缩到 [d,proj]，砍掉 2 个大 GEMM + 2 次拷贝；**实测无条件套用 34/50 失败** ⇒ dtype 分派（fp32 走重排 / 低精度走原路径逐位对齐） | **1.494** | ✅（fp32 档专用） |
| v4 | **attn 输出 `[B,S,H,hd]` 直写**（kernel o_ptr 按 `stride(0),stride(2),stride(1),stride(3)` 落盘），host 侧 `attn.reshape(M,d)` 免费 | **1.7393** | ✅ |
| v5 | fp32 分派开关变体 | 中性 | 未晋级 |
| v6 | **`_kv_reduce_strided_kernel`**：低精度路径的 k/v 转置拷贝折进专用 reduce kernel（直读 k2d [B*S,H*hd]，kr[b,h,p,hd]=Σ_s k2d·w_kr[p,s]，s 按 stride=D 跨步，每 program 一个 (b,h)）| **1.9921** | ✅（v6 备份留存） |
| v7 | 混合变体 | 中性 | 未晋级 |
| v8 | qkv 三输出融合（BM=BN=64 塞 3 份 fp32 acc） | 1.388 | ❌ −30% |
| v9 / v10 | GROUP_M=8 / 2 | 1.9836 / 1.9921 | ≈v6，噪声 |
| v12 | k/v 双输出融合（BM=BN=128, BK=64） | 1.5039 | ❌ −24% |
| v13 | attention dot 换 fp16 操作数（fp32 acc，精度 50/50 过） | 1.8946 | ❌ −5%（小 tile latency-bound） |
| v14 | num_stages 流水 | 无变化 | ❌ |
| v15 | kv_reduce BLOCK_S 64→128 | 1.9992（kernel −16%） | 差一步 |
| **v15b** | **kv_reduce BLOCK_S 128→256**（UB 账 128KB<192KB）| **2.0119–2.0315，50/50** | ✅ **生产版，破 2.0×** |
| v16 | 投影 GEMM BK 低精度档加 K≥256→BK=256 | （未晋级） | 备份 .pyc 留存 |

### v15b 之后的两件大事

1. **2026-08-14 口径复测 = 2.3747×**：benchmark.py 更新后 framework 基线慢 3 倍
   （0.23 vs 0.078ms），同一份代码逐 case 几何平均 2.3747（32/50 >2.0）。
   复现工件原在 `repro_v15b/`（同样被 8/14 清理波及，仅存 anchor 与 .pyc）。
2. **2026-08-14 09:17 清理误删**：v3–v16 的 .py / perf_result.json / report.md 与生产
   `15_LinformerAttention.py` 全部丢失；v15b 仅存 .pyc（BLOCK_S=256 已反汇编核实），
   数字仅存 transcript。**教训：突破版本立即异地备份源码+数据。**

---

## 已排除方向（两条线合并，全部有实测证据）

| 方向 | 结论 | 证据 |
|---|---|---|
| host-concat / launch-fused 投影 | **反模式** | R5 opt_iter_0（0.973 + 9/50 漂移）、R6 opt_iter_0（0.9387 全 case 变慢） |
| 通用 GEMM kernel 吃 4-D strided 布局 | **灾难** | R7 probe：大 hd case 慢 1.86–11.78x |
| reduce 融进 attention（M=hd 小 GEMM） | ❌ | R10 opt_iter_0：0.8882，attention 占 88% |
| W^T kernel 内 swapped-stride | ❌（对上下文敏感） | R10 iter_0：0.0525；预转置后 26x 回升 |
| 多输出 GEMM 融合（UB 逼 tile 减半） | **本芯片死路** | v8 −30%、v12 −24% |
| attention fp16 dot | ❌ | v13 −5%：小 tile latency-bound |
| online softmax（P 单 tile 放得下时） | ❌ | R3：0.5865 |
| 收紧 BK=128 守卫 | ❌ | R5 opt_iter_2：1.3541 |
| GROUP_M 离开 4 | 无效 | v9/v10 |
| num_stages 流水 | 无效 | v14 |
| host 侧 wall-clock 计时定位瓶颈 | **结论全错** | 每 op 摊成常数 ~0.25ms，误判 launch-bound |

## 环境与口径备注

- NPU 自动选卡（config `npu_device_id: null`），R8/R9 记录"logical dev 0 = physical NPU 2"；
  人工线 verify 用过 NPU 4（空闲卡）。共享机器，跨时段数字以同轮 baseline 锚定为可信。
- GLM 网关必须 `ANTHROPIC_MODEL=glm-5.2` 启动（残留 kimi-k2.7-code 会 400）。
- 编排器工件命名：`opt-round-N/output/opt_iter_N/optimized_code.py`（不是 *generated*.py）。
- `freeze_baseline` 锚（`.baseline_anchor.json`）冻结 torch 参考哈希
  `93a543c…`（golden 未被改过，成绩可信）。
