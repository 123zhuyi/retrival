# 99_BlockSparseAttnFwd 迭代日志

环境：CANN 9.1.0 runtime + triton 3.2.0 / triton_ascend 3.2.2（自带 bishengir-compile 1.2.0，
优先级高于 PATH 上的 CANN bishengir）。910B2C，24 AIC / 48 AIV，UB 192KB。
判据：`scripts/verify.py` 50/50 才进 `scripts/benchmark.py --keep_res`（warmup 5 / repeats 20）。

| tag | 改动 | 依据 | verify | speedup | 判定 |
|---|---|---|---|---|---|
| base | 交付基线 `99_BlockSparseAttnFwd_generated.py` 原样 | — | 50/50 | **2.7285** | 基线（历史记录 2.8208，同一量级） |
| v1_tile(初版) | FlashAttention 分块重写 | 见下 | **0/50** | — | 回退：两个 bug |
| v1_tile(修复后) | 同上 + 2 处修复 | 见下 | 50/50 | **71.6328** | **采纳** |

## v1_tile 做了什么

基线 `grid=(total_q, H)`，每个 program 只处理 **1 个 query token**，用
`tl.sum(q[None,:]*k, axis=1)` 手写 fp32 点积（Cube 完全闲置），且每个 token 都要
重新搬一遍整块 K/V。最大 case 是 `total_q=1765, H=128` → 22.6 万个 program。

改成标准 FlashAttention 分块：

1. 任务粒度 `(batch, q_tile=128 行, head)`；`grid=(min(24, total_tasks),)`，
   核内 `for task in range(pid, total_tasks, NUM_CORES)`，NUM_CORES 与 grid 一致。
2. QK^T / PV 都走 `tl.dot`，操作数保持原生 bf16/fp16、累加器 fp32
   （**不**把 q/k 升 fp32——那是过度对齐）。
3. 列块选择表在 **CPU tensor** 上向量化构造（dense / blocksparse / streaming 三类
   头分组批量赋值 + causal 块级预剪枝 + argsort 紧致化），构造完才 `.to(device)`。
   CPU 上的 torch op 不产生 NPU kernel，不进 benchmark 的 operators 统计。
4. 掩码合并：causal 上界与 sk 边界压成同一个 `[BM]` 向量 `lim`，只留一次
   `[BM,BN]` 比较（原写法 `n_valid & causal` 会物化 3 个 [BM,BN] int32 广播临时量）。
5. 整行全掩码用"masked score 填 -3e38 + m 下钳到 -1e30"处理，省掉第二次
   `tl.where(keep, p, 0)`；末尾 `acc / max(l, 1e-30)` 省掉一次 [BM,D] 的 where。
6. CV 重排（17-flash-attention-migration Diff 7）：`tl.dot(p,v)` 提前到
   `alpha`/`p_sum` 之前发射。

## 失败轮与被证伪的假设

### 轮 A：初版 0/50（31 个 AccuracyError + 19 个 NaN 位置断言）
- **假设**：以为是 P 降精度导致的误差。
- **实际**：`matched_ratio=0.558 / max_abs_diff=5.0 / MERE=0.43` —— 不是精度问题，是
  **块表行粒度和 kernel 的 BLOCK_M 不一致**。块表按 `BLOCK_MASK=128` 建行，
  kernel 里 `m0 = qb * BLOCK_M`（当时 BLOCK_M=64/32），于是每个 128 行组只有
  第一个 tile 被算，其余行留在 `torch.empty` 的未初始化内存里 —— 既错值又出 NaN，
  正好同时触发精度判定和 NaN 位置断言。
- **教训**：`torch.empty` 输出 + 部分覆盖 = 未初始化内存，错误形态会伪装成精度问题。

### 轮 B：P 单段降精度
- 修完行粒度后仍 rel 2~4%（bf16 阈值 7.8e-3）。
- **根因**：`p` 是真 fp32 量，`p.to(bf16)` 的 2^-9 相对误差在 PV 里被
  `Σp|v| / |Σpv|` 放大约 20 倍。
- **解法**：hi+lo 二段拆分 `ph=p.to(DTYPE); pl=(p-ph.to(f32)).to(DTYPE);
  pv=dot(ph,v)+dot(pl,v)`。修后 MERE 从 0.43 → 1e-6，max_abs 3.9e-3（=最终 bf16 舍入）。

### 轮 C：tile 可行域（编译器版本相关，历史结论作废）
- 在**错误的** CANN 8.5.1 + bishengir 0.1.0 下：`--enable-ubuf-saving` 直接
  `Unknown command line argument`，D=128 时只有 **32×32** 能编过，
  D=64 只到 64×32（`_logs/scan99_*.log`）。
- 换到 triton_ascend 3.2.2 自带的 bishengir 1.2.0 + `enable_ubuf_saving=True`：
  **128×128 在 D=32/64/128 全部编过**（`_logs/scan99b_*.log`）。
- 128×128 正好等于掩码块粒度，掩码行不用再拆 → 最终定稿用 128×128。
- **结论**：tile 可行域由 UB 预算决定，UB 预算随编译器版本变，**每次换环境必须重扫**。

## 定稿性能

- impl 平均时延 2.2833 ms → **0.0868 ms**（26.3×）
- 几何平均 speedup **71.63x**，50 个 case 无一回退
- 瓶颈画像（主导 case）：`aic_scalar_ratio=0.485`、`aiv_vec_ratio=0.463`、
  `aic_mac_ratio=0.099`、串行度 (aic+aiv)/Duration = **1.66**（> 1.5，Cube-Vector 串行）
  → 单 kernel 已经不是访存瓶颈，剩余空间在 CV 流水与同步。
