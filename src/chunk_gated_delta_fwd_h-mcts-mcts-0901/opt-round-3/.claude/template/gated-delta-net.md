---
name: gated-delta-net
description: GDN（Gated DeltaNet）类算子（OlmoHybridGdnFullForward / Gdn2Fwd / ChunkGatedDeltaRule / FusedGdnGatingKernel）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# GDN（Gated DeltaNet）类算子优化经验

本文档合并该类别算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的递归 + 门控约束（已提取，各算子章节不再重复；与其他类别共用的通用约束见 `tensor-transform.md` G1-G8 / `recurrent-neural-network.md` R1-R5 / `math-compute.md` M1-M8 等，此处只写本类别特有的）
- **§2 OlmoHybridGdnFullForward**（gdn-full-fwd，首个归档，最完整 6-kernel 参考）
- **§3 Gdn2Fwd**（gdn2-fwd，单 kernel 完整前向 + varlen）
- **§4 ChunkGatedDeltaRule**（chunked-gdr，分块 GDR）
- **§5 FusedGdnGatingKernel**（gdn-gating，门控片段）
- **§6 常见陷阱**（按算子分小节）

> ⚠️ **关键区分**：本类别的核心优化哲学是 **「状态矩阵 S[K,V] 按 (batch,head) 持全 K 维、V 维 tile 分核 + kernel 内 `for t` 串行递归 + 全程 fp32 累加 + 数值稳定门控」**。GDN 是"逐时间步外积写状态 + 门控衰减"的递归算子，**不是** attention（禁止把时间维当 seq 维做并行扫描）、**不是**普通 RNN 的 gate-multiplicative 递归（R2/R4/R5 不适用，见 D1 差异）。生成时**禁止混用** attention/GEMM 类别的并行经验。
> 首次归档算子：`68_OlmoHybridGdnFullForward`（单层 GDN 完整前向，50-case 多 shape，ascend910_9372）；归档时最佳几何平均 **12.479x**。
> 历史经验仅供启发，**禁止直接复制代码结构**；变量名/结构必须重新设计。

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| OlmoHybridGdnFullForward | `gated-delta-net` | conv1d + GDN 门控 + `h_t = g_t·h_{t-1} + k_t·v_t^T` + gated RMSNorm + 输出投影 | 6 kernel 分解，recurrence fp32 串行 + 按 head/V-tile 分核 |
| Gdn2Fwd | `gated-delta-net` | `S_t = exp(g)·S_{t-1} + k·(w·v − b·S)` 带 erase 项 + 可选 QK L2 norm + varlen | 单 kernel 完整前向，V_TILE 分核 + 编译期 flag |
| ChunkGatedDeltaRule | `gated-delta-net` | 分块 delta rule，同 Gdn2Fwd 递归 + 终态 2D store | V-tile 分核 + 编译期 flag + `multibuffer/unit_flag`（基准修正后 **25.57x**，精度 24/50 未达标） |
| FusedGdnGatingKernel | `gated-delta-net` | GDN 门控片段：`gate = −exp(A_log)·softplus(x·beta)/beta` + `beta=sigmoid(b)` | 单 kernel 纯 elementwise，Taylor log1p + threshold 稳定 |

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下 8 条约束是本类别算子**共有**且**未在其他 template 文件覆盖**的工程约束。其他文件已提取的通用约束（如 `tensor-transform.md` 的 G1 动态 num_cores / G2 pow2 BLOCK / G4 grid 不超核数 / G7 contiguous 等）此处不再重复，各算子章节引用时标注。

### D1 状态矩阵 S[K,V]：每个 program 必须持全 K 维，V 维 tile 分核
- **必须** 让状态矩阵 `S`（68 里记作 `h`，[K,V]，K=key_dim、V=value_dim）**整体驻留在单个 program**：K 维以 `tl.arange(0, K_PAD)` 完整持有，V 维通过 `program_id` 切 `V_TILE`；grid 落在 `(batch · head · num_v_tiles)`。**禁止**把 K 维切成多个 program 或按 K 维分核"提升并行度"。
- **Why:** 递归更新是**外积** `S_t = decay·S_{t-1} + k_t[:,None]·v_new[None,:]`，`k_t` 是长度 K 的向量、`v_t` 长度 V。归约维是 K：`S` 的每一列（一个 V）都依赖全部 K。跨 program 切 K → 每个 program 只有半段 S，下一步归约读全 K 时 RAW 冒险。而 V 维各列相互独立（`v_new[None,:]` 按列广播），可安全分核。
- **How to apply:** kernel 内 `S = tl.zeros((BLOCK_K, V_TILE), dtype=tl.float32)`（或从 initial_state load），主循环内 `S = S * decay[:,None] + k_t[:,None] * v_new[None,:]`。
- **典型应用**：68 gdn_recurrence（grid `(HV, HVD/bv)`）、94 _gdn2_full_kernel（grid `(B·H·num_v_tiles)`）、91 cgdr（grid `(B·HV·V_TILES)`）。
- 与 `recurrent-neural-network.md` R2（每 program 持完整隐维 H）的差异：RNN 的归约是 `h_{t-1}@W_hh` 矩阵乘（全 H 读全 H）；GDN 是外积写状态，**V 维天然可分核**，因此 GDN 能比 RNN 多用一维并行度（batch×head×V_tile），这是 GDN 性能远好于 RNN 的原因之一（68: 12.5x vs LSTM: 13.4x 且 GDN 无 host-loop 必需性）。

### D2 时间维必须在 kernel 内 `for t` 串行，decay 门控逐步乘进 S
- **必须** 在单个 program 内 `for t in range(T_start, T_end)` 串行推进时间步；decay/gate 每步 `S = S * exp(g_t)[:,None]` 先衰减再加外积。**禁止**把时间维展平进 grid 或跨 program 并行。
- **Why:** `h_t = g_t·h_{t-1} + k_t·v_t^T` 是真串行依赖。**与 RNN 不同，GDN 不需要 host-loop 拆 kernel**：68/94 在 kernel 内 `for t` 且 50/50 稳定（冷启动无 LSTM 那种首启异常），91 同样结构但精度 24/50 未达标（见 §4 L1.1 —— 是 verify/数值问题，非结构问题），因为 GDN 的单步计算全是向量外积 + elementwise，没有 RNN 那种"每步一个 K-loop 小 GEMM 的复杂地址算术"。只有性能优化阶段才考虑把 GEMM 段（68 的输出投影）拆出去。
- **典型应用**：68 gdn_recurrence、94 _gdn2_full_kernel、91 cgdr 全部。

### D3 递归状态与输出投影必须 fp32 累加（精度硬前提）
- **必须** 状态矩阵 `S` 全程保持 `tl.float32`，只在最终写 `o` / `final_state` 时 `.to(out_dtype)`。输出投影若存在（`h @ W_o`），用 `acc = tl.zeros(fp32)` + `tl.dot(..., out_dtype=tl.float32)`。**禁止** fp16/bf16 累加递归。
- **Why:** 精度阈值 MERE 要求。68 实测 fp16 累加变体 verify 27/50，退回 fp32 → 50/50（9 轮精度迭代中 fp16 探索明确证伪）。递归误差随时间步累积，低精度必然超标。
- **典型应用**：68 gdn_recurrence + linear_fp32、94（`S_work` 显式 `to(torch.float32)`）、91（`state = tl.zeros(..., dtype=tl.float32)`）。
- 与 `gqa.md`（fp32 dot 逐位一致）的差异：GDN 递归必须**全程** fp32，不是只在 dot 上。

### D4 门控/归一化数值稳定：Taylor log1p + threshold + rsqrt norm
- **必须**（三处稳定化）：
  1. softplus 类：`sp = where(y<0.2, _taylor_log1p(y), log1p(y))`，再做 `where(t<=THRESHOLD, sp, x)` 兜底大值分支（防止 `exp(x·beta)` 溢出为 inf）；91 的简化版 `where(gi>20.0, gi, log(1+exp(gi)))` 同理。
  2. QK L2 norm：`k_inv = tl.rsqrt(tl.sum(k*k) + EPS)` 再乘回，**禁止** `k / sqrt(sum)`（除零 + 溢出），EPS=1e-6。
  3. `exp` 之后接 `where` 裁剪巨值门控（`gate = -exp(A_log)·sp` 中 `exp(A_log)` 可能巨大）。
- **Why:** GDN 门控是 `exp` 族（`g = exp(g_t)`、`gate = -exp(A_log)·softplus`）。输入范围宽时 `exp` 溢出 → inf/nan 污染整个递归链；实测 softplus 小值区用 Taylor（8 项）比 log1p 更快且精度够。
- **How to apply:** 82 的 `_taylor_log1p` 是 8 项多项式（y 到 y^8），小值区（|y|<0.2）替代 `log1p`；`BETA`/`THRESHOLD` 作为 `tl.constexpr` 编译期下发。
- **典型应用**：82 fused_gdn_gating（全部 3 处）、91/94 USE_QK_L2NORM。

### D5 S 矩阵 tile 面积受 UB 约束（BLOCK_K × BLOCK_V ≲ 1024）
- **必须** `BLOCK_K × BLOCK_V` 面积不超过 UB 承载量（实测上限 ~1024 元素/tile 量级）；91 用 `BLOCK_V = 1024 // BLOCK_K` 再 cap 到 128/64。
- **Why:** 91 注释明确"终态 2D store staging 使 UB 需求 ~1100 bits/元素，tile 面积压到 ≤1024"。S 是 [K,V] 2D 驻留 + 每步外积写，UB 压力随面积线性涨。
- **How to apply:** K 大时 V_TILE 自动缩小：`num_v_tiles = ceil(V / V_TILE)`；K_PAD 用 `next_pow2(K)` 但**不**让 K_PAD×V_TILE 超限。
- **典型应用**：91（BLOCK_V 分档）、94（V_TILE=64 当 V>64，K_PAD 独立）。
- 与 `math-compute.md` M4（UB 决定 tile）的差异：GDN 的 UB 杠杆是**状态 tile 面积**，不是 GEMM 的 BM/BN/BK。

### D6 分核 tile 必须 `min(tile, dim)` 保护，防 grid=0 编译崩
- **必须** 所有按维度分核的 tile 在 host 侧做 `tile = min(tile, dim)`，grid 用 `ceil(dim/tile)`；kernel 内 `mask` 兜底越界。
- **Why:** 68 族B（HVD=64）若固定 bv=128 → `HVD // bv = 0` → grid=0 → 编译期崩溃。94 的写法是 `V_TILE = next_pow2(V) if V<=64 else 64`。
- **典型应用**：68 gdn_recurrence 的 `bv = min(bv, head_v_dim)`（GA 基因设计时踩过）、94 的 V_TILE 分档。

### D7 GDN kernel 启动默认带 `multibuffer=True, unit_flag=True`
- **必须** 所有 GDN 递归 kernel（含依赖 V-tile 分核的）在 host 启动时带 `multibuffer=True, unit_flag=True`；**禁止**省略这两个 Triton-Ascend 编译选项。
- **Why:** 91 实测：同一 `_cgdr_fwd_kernel`，加了这两个选项 impl 从 **1083us → 69.8us（15.5x）**，是 25.57x 的两大功臣之一。它们让编译期启用双缓冲 + 单元标志，显著降低递归 kernel 的流水气泡。
- **How to apply:**
  ```python
  _cgdr_fwd_kernel[grid](..., multibuffer=True, unit_flag=True)
  ```
- **典型应用**：91 cgdr（实测 25.57x）；68 gdn_recurrence / 94 _gdn2_full_kernel 若未带应补上（潜在 +Xx）。
- 与 `recurrent-neural-network.md` R5 的差异：R5 是"少 load/地址算术"；D7 是编译选项层（无需改 kernel 代码）。

### D8 看到 <1x speedup 先查基准污染，再决定是否动结构
- **必须** benchmark 出现 <1x 时，先核对 `perf_result.json` 的 `framework.avg_latency_ms` 是否量级合理（与独立复验跑对比），**确认基准真实后再谈 kernel 结构优化**。**禁止**直接归咎"kernel 全串行/结构错"而推翻已有正确实现。
- **Why:** 91 的 0.135x 实为基准污染：同一 torch 基准在 `output/perf_result.json` 测出 framework **0.0379ms**，在 `manual_iter/optimized_perf.json` 测出 **1.322ms**（差 35x），而 impl kernel 几乎逐字节相同。0.135x 是假象，真实 speedup 25.57x。
- **How to apply:** 用 `optimized_perf_npu2.json` 这类复验跑交叉验证；比对 `framework.avg_latency_ms` 而非只信一个 speedup 数字。
- **典型应用**：所有 GDN 算子；91 是最完整的 case study。

---

## §2 OlmoHybridGdnFullForward 算子（gdn-full-fwd）

**算子类别**: `gated-delta-net`
**典型特征**: 单层 GDN 完整前向（Olmo Hybrid）：causal depthwise conv1d(q,k,v)+SiLU → GDN 门控 → 预归一化 → GDN 递归 → gated RMSNorm → 输出投影。输入 `x[T,C]`，50 case 三族 shape（HK=16/30, HKD=96/128, HVD=64/128, GQA_REP=1/2），dtype 约半 bf16/fp16，T∈[1,1024]。
**性能基准**: 50/50 pass，几何平均加速比 **12.479x** vs torch（ascend910_9372，纯 Triton 交付）

### §2.0 首次生成必读：一次把 6-kernel 框架写对

本算子 4/50 → 50/50 经历了 **9 次 verify 迭代**（详见 GQA收益分析 迭代日志）。绝大多数失败不是算法错而是**框架错**：recurrence 用了 fp16 累加、bv 没带 `min(bv, HVD)` 保护、kernel 内仍有 Python 串行循环。**首次生成就必须**：6 kernel 全 Triton、recurrence fp32、`bv=min(bv,head_v_dim)`、conv 用二维 tile。框架对了，精度一次过，提速自然来。

### §2.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 6 kernel 独立分解，各司其职不融合
- **必须** 分解为：`causal_conv1d_silu`（深度卷积+SiLU）/ `gdn_gating`（GDN 门控 g、β）/ `compute_norms`（k_norm、q_norm 预计算）/ `gdn_recurrence`（串行递归 fp32）/ `gated_rmsnorm`（gated RMSNorm）/ `linear_fp32`（输出投影 GEMM）。
- **禁止** 把 gating/norms/rmsnorm 融合进 recurrence（各 <5% 耗时，融合只会加大 recurrence 的 UB 压力）。
- **Why:** kernel 分解实测 conv/recurrence/linear 三 kernel 合计占 85-98%，其余三 kernel 各 <5%，融合无收益且抬高 recurrence 复杂度。

#### L1.2 recurrence 必须 fp32 + `bv=min(bv, head_v_dim)`
- **必须** `gdn_recurrence_kernel` 的 V-tile `bv = min(bv, head_v_dim)`；状态 `h` 用 `tl.float32`。
- **禁止** 固定 bv 不做 dim 保护（族B HVD=64 → grid=0 编译崩）；**禁止** fp16 累加。
- **Why:** 见 D3/D6。实测族B（HVD=64）是最容易被 GA 基因设计踩崩的形状。

#### L1.3 输出投影 GEMM 用 fp32 累加（对齐 golden）
- **必须** `linear_fp32_kernel`：`acc = tl.zeros(acc, tl.float32)` + `tl.dot(a, b, out_dtype=tl.float32)`，tile `BLOCK_M/N/K=64/128/64, num_warps=4`。
- **禁止** 低精度 dot（fp16 累加）。
- **Why:** golden 是 `F.linear(x, w_o_proj.float())`，低精度 dot 精度超标。**性能代价极小**（CANN `npu_linear` 仅 +1.19x，为纯 Triton 合规放弃）。

#### L1.4 conv 二维 tile（BLOCK_T × BLOCK_C）
- **必须** `causal_conv1d_silu_kernel` 用 `BLOCK_T=32, BLOCK_C=128`，grid `(T/32, conv_dim/128)`，每个 program 处理一段 T × 一段通道的独立深度卷积。
- **禁止** 沿 T 串行访问通道维。
- **Why:** 二维 tile 让 conv 占比从 >50% 降到 26-46%（case16/22 数据）。

### §2.2 Layer 2: 算法骨架

```
# host 侧（ModelNew.forward）
# x[T,C] --Wqkv--> projected[T, 2*K+V+gate]；切分 q[T,HK,HKD] k[T,HK,HKD] v[T,HV,HVD] gate[T,HK]
1. causal_conv1d_silu_kernel[grid_conv](q,k,v → q_c,k_c,v_c)        # 深度卷积 + SiLU
2. gdn_gating_kernel[grid_gate](gate, dt_bias, A_log → g, β)        # g = sigmoid(gate)·exp(dt_bias·A_log); β=sigmoid(...)
3. compute_norms_kernel[grid_norm](k_c, q_c, running_stats → k_norm, q_norm)  # k·rms / q·rms
4. gdn_recurrence_kernel[grid_rec](q_norm, k_norm, v_c, g → h[T,HV,HVD])      # h_t = g_t·h_{t-1} + k_t·v_t^T, fp32, bv=min(bv,HVD)
5. gated_rmsnorm_kernel[grid_rms](h, g → out)                        # gated_rmsnorm(h,gate)·sigmoid(gate)
6. linear_fp32_kernel[grid_lin](out, W_o → output)                   # output = out @ W_o^T, fp32 dot
```

### §2.3 Layer 3: 关键 kernel 实现（优化重点与易错代码）

#### L3.1 `gdn_recurrence_kernel`

```python
@triton.jit
def gdn_recurrence_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, o_ptr,
    T, scale,
    HK, HVD, H,          # head dims
    stride_qb, stride_qt, stride_qh, stride_qk,   # q/k/g 共享 q 布局
    stride_vb, stride_vt, stride_vh, stride_vv,
    stride_ob, stride_ot, stride_oh, stride_ov,
    bv: tl.constexpr, K_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    v_tile_idx = pid % (HVD // bv)
    bh = pid // (HVD // bv)
    h = bh % H
    b = bh // H

    k_offs = tl.arange(0, K_PAD)
    k_mask = k_offs < HK
    v_offs = v_tile_idx * bv + tl.arange(0, bv)
    v_mask = v_offs < HVD

    acc = tl.zeros((K_PAD, bv), dtype=tl.float32)      # S[h]: [K_PAD, bv]，持全 K，V 只切 bv
    for t in range(T):
        g_t = tl.load(g_ptr + b*stride_qb + h*stride_qh + t*stride_qt + k_offs*stride_qk, mask=k_mask, other=0.0)
        g_t = tl.exp(g_t)                               # decay 门控
        acc = acc * g_t[:, None]
        k_t = tl.load(k_ptr + ..., mask=k_mask, other=0.0).to(tl.float32)
        v_t = tl.load(v_ptr + ..., mask=v_mask, other=0.0).to(tl.float32)
        acc = acc + k_t[:, None] * v_t[None, :]         # 外积写状态（fp32）
        q_t = tl.load(q_ptr + ..., mask=k_mask, other=0.0).to(tl.float32)
        o_t = tl.sum(q_t[:, None] * acc, axis=0) * scale # q^T·S，归约沿 K
        tl.store(o_ptr + ..., o_t.to(o_ptr.dtype.element_ty), mask=v_mask)
```

**host 预计算**：`bv = min(BV_CONFIG, head_v_dim)`（`head_v_dim=HVD` 取实际值，族B=64 时必须回落），`grid = (HV, HVD//bv)`，`K_PAD = next_pow2(HK)`。

#### L3.2 `linear_fp32_kernel`（输出投影）

```python
@triton.jit
def linear_fp32_kernel(in_ptr, w_ptr, out_ptr, M, N, K,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    pm = pid // (tl.cdiv(N, BN)); pn = pid % (tl.cdiv(N, BN))
    rm = pm*BM + tl.arange(0, BM); rn = pn*BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)          # fp32 累加对齐 F.linear(x, W.float())
    for k in range(0, K, BK):
        a = tl.load(in_ptr + rm[:,None]*K + (rk[None,:]+k), mask=(rk[None,:]+k) < K, other=0.0)
        w = tl.load(w_ptr + (rk[:,None]+k)*N + rn[None,:], mask=(rk[:,None]+k) < K, other=0.0)
        acc += tl.dot(a, w, out_dtype=tl.float32)
    tl.store(out_ptr + rm[:,None]*N + rn[None,:], acc.to(out_ptr.dtype.element_ty), mask=(rm[:,None]<M)&(rn[None,:]<N))
```

### §2.4 OlmoHybridGdnFullForward 性能基准

| shape 族 | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| 族A（HK=30/HVD=128/conv=9600） | case1-11,45-49 | ~5.8 ~ 38.9x | T 大时 conv+recurrence 主导 |
| 族B（HK=16/HVD=64/conv=6144） | case12-22 | ~2.6x（case23 T=1）~ 15x | HVD=64，bv 必须回落 |
| 族C（HK=16/HVD=128/conv=8192） | case23-33,50 | 2.55x（T=1）~ 38.9x（T=1024） | |
| **全量** | **50** | **12.479x** | 50/50 全过，纯 Triton |

**关键结论**:
1. **T 小 → GEMM + launch 主导**（T=8 时 linear 52.7%），硬下限 ~2.5x（T=1 的 launch 开销）。
2. **T 大 → recurrence 串行递归主导**（T=768 时 recurrence 46.5%），随 T 线性增长，是**结构性硬上限**，tile 优化只能减常数。
3. 混合版（CANN `npu_linear` 输出投影）达 13.669x，纯 Triton 损失仅 1.19x —— 交付选纯 Triton。

---

## §3 Gdn2Fwd 算子（gdn2-fwd）

**算子类别**: `gated-delta-net`
**典型特征**: GDN2 完整前向（带 erase 项的 delta rule）：`S_t = exp(g)·S_{t-1} + k·(w·v − b·S)`，可选 QK L2 norm，支持 equal_length 与 varlen（cu_seqlens）+ initial/final_state。输入 q/k/v [B,T,H,K]（v [B,T,HV,V]），单 kernel 完成全部。
**性能基准**: 50/50 pass，几何平均加速比 **2.5858x** vs torch

### §3.1 Layer 1: 设计约束（Gdn2Fwd 专属）

#### L1.1 单 kernel 完整前向 + 编译期 flag 控制变体
- **必须** 用 1 个 `_gdn2_full_kernel` 完成"load S → for t → 每步 g/erase/外积/输出"，变体（QK L2 norm、门控形式）用 `tl.constexpr` 编译期开关：`USE_L2NORM: tl.constexpr`。
- **禁止** 为每个变体写独立 kernel。
- **Why:** 变体是编译期可消的，constexpr 分支零运行时开销；host 侧只透传 bool。

#### L1.2 erase 项（b_t·S）必须先算再外积
- **必须** 每步顺序：`bk = b_t·k_t` → `erase = sum(bk[:,None]·S, axis=0)` → `v_new = w_t·v_t − erase` → `S = S + k_t[:,None]·v_new[None,:]`。
- **禁止** 先做 `S·exp(g)` 后忘记 erase 的 w·v 分量。
- **Why:** 这是 GDN2 与基础 GDN 的差异点：多一个**沿 K 的 erase 归约**（`sum(bk·S)`），本质仍归约 K 维，必须在单 program 内完成（D1 同源约束）。

#### L1.3 varlen / initial_state / final_state 三态路径
- **必须** `cu_seqlens` 非空时逐 segment 启动 kernel（`for n in range(N)`，每段 `S_work` 从 initial_state 或 zeros 初始化，算完写 final_state[n]）；equal-length 时一次启动全 B。
- **禁止** 把 varlen 的段内时间偏移写死在 kernel 里。
- **Why:** 输入 API 显式要求 varlen + 状态传递；段间状态独立，逐段启动即可（每段 T 短，launch 成本可控）。

### §3.2 Layer 2: 算法骨架

```python
def forward(q, k, v, g, b, w, initial_state, cu_seqlens, scale, use_qk_l2norm_in_kernel):
    K_PAD = _next_pow2(K)
    V_TILE = _next_pow2(V) if V <= 64 else 64        # V>64 分块，K_PAD×V_TILE 面积受控
    num_v_tiles = (V + V_TILE - 1) // V_TILE
    if cu_seqlens:                                    # varlen 路径
        grid = (H * num_v_tiles,)
        for n in range(N):
            s, e = cu_seqlens[n], cu_seqlens[n+1]
            S_work = initial_state[n:n+1].float() if initial_state else zeros(1,H,K,V,float)
            _gdn2_full_kernel[grid](..., S_work, o, s, e, ...)   # 段内 for t in range(s,e)
            final_state[n] = S_work[0]
    else:                                             # equal-length 路径
        grid = (B * H * num_v_tiles,)
        _gdn2_full_kernel[grid](..., S_work, o, 0, T, ...)
```

### §3.3 Layer 3: 关键 kernel 实现

#### L3.1 `_gdn2_full_kernel`（核心递归段）

```python
@triton.jit
def _gdn2_full_kernel(q_ptr, k_ptr, v_ptr, g_ptr, b_ptr, w_ptr, S_ptr, o_ptr,
                      T_start, T_end, scale, B, H, K, V, ...strides...,
                      K_PAD: tl.constexpr, V_TILE: tl.constexpr,
                      num_v_tiles: tl.constexpr, USE_L2NORM: tl.constexpr, EPS: tl.constexpr):
    pid = tl.program_id(0)
    v_tile_idx = pid % num_v_tiles
    bh = pid // num_v_tiles
    h = bh % H; b = bh // H
    k_offs = tl.arange(0, K_PAD); k_mask = k_offs < K
    v_offs = v_tile_idx * V_TILE + tl.arange(0, V_TILE); v_mask = v_offs < V

    S = tl.load(S_ptr + ... + k_offs[:,None]*sk + v_offs[None,:]*sv,
                mask=k_mask[:,None] & v_mask[None,:], other=0.0)   # fp32 状态

    for t in range(T_start, T_end):
        g_t = tl.load(g_ptr + ...).to(tl.float32)
        exp_g = tl.exp(g_t)
        S = S * exp_g[:, None]                                     # decay

        k_t = tl.load(k_ptr + ...).to(tl.float32)
        b_t = tl.load(b_ptr + ...).to(tl.float32)
        if USE_L2NORM:                                             # D4: rsqrt 稳定 norm
            k_t = k_t * tl.rsqrt(tl.sum(k_t*k_t) + EPS)
        erase = tl.sum((b_t*k_t)[:, None] * S, axis=0)             # 沿 K 归约 erase
        v_t = tl.load(v_ptr + ...).to(tl.float32)
        w_t = tl.load(w_ptr + ...).to(tl.float32)
        S = S + k_t[:, None] * (w_t*v_t - erase)[None, :]          # 外积写状态

        q_t = tl.load(q_ptr + ...).to(tl.float32)
        if USE_L2NORM:
            q_t = q_t * tl.rsqrt(tl.sum(q_t*q_t) + EPS)
        o = tl.sum((scale*q_t)[:, None] * S, axis=0)               # q^T·S
        tl.store(o_ptr + ..., o.to(o_ptr.dtype.element_ty), mask=v_mask)

    tl.store(S_ptr + ..., S, mask=k_mask[:,None] & v_mask[None,:]) # final_state 回写
```

### §3.4 Gdn2Fwd 性能基准

| 项 | 值 |
|------|-------|
| 全量 | **2.5858x** / 50/50 |
| 主结构 | 单 kernel for t 串行，V_TILE=64 分块 |
| 优化空间 | 未做 tile 扫描；recurrence 仍串行（D2），GEMM 未拆分 |

**关键结论**: 单 kernel 完整前向正确性已达标（50/50），但 2.59x 远低于 68 的 12.5x —— 因为 GDN2 多了 erase 项（每步多一次 `sum(bk·S)` 归约 + `w·v`）且没做 conv/GEMM 拆分。**优化方向**：把纯 elementwise 段（gate/norm/erase 预计算）拆成独立 kernel，recurrence 只留外积写状态；仿 68 的 6-kernel 结构。

---

## §4 ChunkGatedDeltaRule 算子（chunked-gdr）

**算子类别**: `gated-delta-net`
**典型特征**: Chunked Gated Delta Rule 前向（同 Gdn2Fwd 的递归式，另含 chunk_size / use_bias / use_initial_state / allow_neg_eigval / state_v_first 等超参），输入 q/k/v [B,T,H,K]，v [B,T,HV,V]，HV=R·H。
**性能基准**: 50/50 pass（早期 `output/opt_iter_*`）；后期 `manual_iter` 实测 **25.5704x** vs torch（avg 69.8us，NPU2 复验 23.2953x），但 verify 仅 **24/50** 精度未达标。**速度与精度分离：25.57x 不代表可交付，需先修 24→50 再谈提交**。

### §4.1 Layer 1: 设计约束（ChunkedGatedDeltaRule 专属）

#### L1.1 早期"0.135x"结论要修正：那是基准污染，不是结构天花板
- **必须** 先核对 benchmark 的 torch 基准是否被**快速路径污染**再谈优化：91 的 `output/perf_result.json`（0.1351x）framework avg 仅 **0.0379ms**，而 `manual_iter/optimized_perf.json`（25.57x）framework avg **1.322ms** —— 同一 torch 基准、同一定义文件（sha256 相同），torch 侧却差 **35x**。0.1351x 是基准异常（torch 侧走了极快路径），**不是 impl 慢 185x**。
- **禁止** 在看到 <1x speedup 时直接归咎于"kernel 全串行"而推翻结构。先查 `perf_result.json` 的 `framework.avg_latency_ms` 是否量级合理（与 `optimized_perf.json` 的 framework 对比）。
- **Why:** 91 的 impl kernel 在两个版本里几乎逐字节相同（仅差 clamp 方式 + `multibuffer/unit_flag`），0.135x 与 25.57x 的差距主要来自基准本身。
- **How to apply:** 用 `optimized_perf_npu2.json` 等**复验跑**核对 speedup，不采信单次 benchmark。

#### L1.2 chunk 并行是**未验证的潜力方向**，不是当前实现的缺陷
- **必须** 认清现状：25.57x 的实现在 kernel 内 `for t in range(T)` **全串行**，`chunk_size` 只是参数透传，**并未做块内并行**——但仍达 25.57x（因为递归本身在 V_TILE 分核 + multibuffer 下已足够快，torch 基准 1.322ms 慢）。
- **禁止** 宣称"必须实现 chunk 并行才能提速"。**这是未验证假设**，潜力方向而非当前瓶颈。
- **How to apply:** 若要试 chunk 并行：`for chunk in range(T//C)`，块内把 [C,K] 的 q/k 一次 load 张量化外积，块间只更新 S。**但要先复现 25.57x 并确认基准真实**，否则无从对比收益。
- 与 `flash_attention.md` 的差异：FA 块间不传状态；GDR 块间有 `exp(g)` 衰减传递，块边界是串行点。

#### L1.3 编译期 flag 全覆盖（变体矩阵）
- **必须** 用 `tl.constexpr` 覆盖：`USE_GATE_IN_KERNEL`（gate 由 A_log+dt_bias 现场算 vs 预计算）、`USE_QK_L2NORM`、`ALLOW_NEG_EIGVAL`（b_t 直接乘 vs sigmoid）、`USE_BETA_SIGMOID`。
- **Why:** 该算子官方 API 超参多，运行时 bool 分支杀性能；constexpr 消分支。
- **gate 现场算**：`sp = where(gi>20.0, gi, log(1+exp(gi)))`（D4 稳定 softplus）；`decay = exp(gate)`。

#### L1.4 STRIDED_STATE 加载必须行连续取向 + 钳位（避 gather 标量化）
- **必须** state_v_first=True（`STRIDED_STATE`）时，按 `[V,K]` 行连续取向加载：`rv = where(v_mask, v_offs, 0)`，`offs2d = rv[:,None]*ss0_v + k_offs[None,:]*ss0_k`，`tl.load(s0_ptr + ... + offs2d)`；只钳行索引，保持行内连续向量 load。**禁止**全偏移 `tl.where(vk_mask, offs2d, 0)` 的掩码钳位（实测导致 gather 标量化 + 507035）。
- **Why:** svf=1 时 [K,V] 块内行跨步是标量化毒药；[V,K] 行连续取向 + 仅钳行索引避免标量化。92 的 `_gdn2_full_kernel` 同源注释也强调这一点。
- **How to apply:** 见 L3.1 的 STRIDED 分支。

#### L1.5 终态 2D store 的 UB staging：`BLOCK_V = 1024 // BLOCK_K`
- **必须** 输出 final_state 时 tile 面积 `BLOCK_K × BLOCK_V ≤ 1024`；非终态路径 `BLOCK_V = min(next_pow2(V), 128)`。
- **Why:** 终态要 2D 逐块 store 回 [B,HV,K,V]，UB 需求 ~1100 bits/元素；面积超限直接 AICore 异常/性能崩。
- **How to apply:** `BLOCK_K = min(next_pow2(K), 128)`；`BLOCK_V = 1024//BLOCK_K if output_final_state else min(next_pow2(V), 128)`。

### §4.2 Layer 2: 算法骨架

```python
def forward(q,k,v,g,beta,A_log,dt_bias,initial_state,scale,chunk_size,
            use_gate_in_kernel,use_bias,use_initial_state,output_final_state,
            use_qk_l2norm,use_beta_sigmoid,allow_neg_eigval,state_v_first):
    BLOCK_K = min(next_pow2(K), 128)
    BLOCK_V = 1024 // BLOCK_K if output_final_state else min(next_pow2(V), 128)
    V_TILES = (V + BLOCK_V - 1) // BLOCK_V
    grid = (B * HV * V_TILES,)                                 # 25x 实测结构
    _cgdr_fwd_kernel[grid](..., BLOCK_K=BLOCK_K, BLOCK_V=BLOCK_V,
                           STRIDED_STATE=bool(state_v_first),
                           multibuffer=True, unit_flag=True)   # ★ 25x 关键：Triton-Ascend 双缓冲/单元编译选项
    # 25x 实现 kernel 内仍 for t in range(T) 全串行；chunk_size 未用于并行
    # state = tl.zeros((BLOCK_K, BLOCK_V), float32)（或从 initial_state load）
    # for t: decay=exp(stable_softplus(g_t)); state=state*decay[:,None];
    #        k_row=l2norm(k_row) if qk_l2norm; v_row=(w·v - b_t·S);
    #        state += k_row[:,None]*v_row[None,:];
    #        o_t = scale*sum(q_row[:,None]*state, axis=0); store(o_t)
```

### §4.3 Layer 3: 关键 kernel 实现

#### L3.1 `_cgdr_fwd_kernel`（递归核心，与 94 同构，25x 实测结构）

```python
@triton.jit
def _cgdr_fwd_kernel(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, alog_ptr, dtb_ptr, s0_ptr,
                     out_ptr, sf_ptr, T, HV, R, K, V, V_TILES, scale, ...strides...,
                     USE_GATE_IN_KERNEL: tl.constexpr, USE_BIAS: tl.constexpr,
                     USE_INITIAL_STATE: tl.constexpr, OUTPUT_FINAL_STATE: tl.constexpr,
                     USE_QK_L2NORM: tl.constexpr, USE_BETA_SIGMOID: tl.constexpr,
                     ALLOW_NEG_EIGVAL: tl.constexpr, BLOCK_K: tl.constexpr,
                     BLOCK_V: tl.constexpr, STRIDED_STATE: tl.constexpr):
    pid = tl.program_id(0)
    i_bh = pid // V_TILES; i_vt = pid - i_bh * V_TILES
    i_b = i_bh // HV; i_hv = i_bh - i_b * HV; i_h = i_hv // R
    k_offs = tl.arange(0, BLOCK_K); v_offs = i_vt*BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = k_offs < K; v_mask = v_offs < V; kv_mask = k_mask[:,None] & v_mask[None,:]
    state = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)    # D1: 持全 K，V tile
    if USE_INITIAL_STATE:
        if STRIDED_STATE:                                     # L1.4: [V,K] 行连续 + 只钳行索引
            rv = tl.where(v_mask, v_offs, 0)
            offs2d = rv[:, None] * ss0_v + k_offs[None, :] * ss0_k
            state = tl.trans(tl.load(s0_ptr + i_b*ss0_b + i_hv*ss0_h + offs2d).to(tl.float32))
        else:
            state = tl.load(s0_ptr + ..., mask=kv_mask, other=0.0).to(tl.float32)
    if USE_GATE_IN_KERNEL:
        alog = tl.load(alog_ptr + i_hv).to(tl.float32); neg_exp_alog = -tl.exp(alog)
        dtb = tl.load(dtb_ptr + i_hv).to(tl.float32) if USE_BIAS else 0.0
    for t in range(T):                                        # 25x 实测：全串行仍达 25.57x
        gi = tl.load(g_base + t*sg_t).to(tl.float32)
        if USE_BIAS: gi = gi + dtb
        sp = tl.where(gi > 20.0, gi, tl.log(1.0 + tl.exp(gi)))   # D4 stable softplus
        gate = neg_exp_alog * sp if USE_GATE_IN_KERNEL else gi
        decay = tl.exp(gate)
        bt = tl.load(b_base + t*sb_t).to(tl.float32)
        if USE_BETA_SIGMOID:
            bt = 1.0/(1.0+tl.exp(-bt))
            if ALLOW_NEG_EIGVAL: bt = bt * 2.0
        k_row = tl.load(k_base + t*sk_t + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        q_row = tl.load(q_base + t*sq_t + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        v_row = tl.load(v_base + t*sv_t + v_offs, mask=v_mask, other=0.0).to(tl.float32)
        if USE_QK_L2NORM:                                     # D4: rsqrt 稳定（勿用 /max(sqrt,1e-6)）
            k_row = k_row * tl.rsqrt(tl.sum(k_row*k_row, axis=0) + 1e-6)
            q_row = q_row * tl.rsqrt(tl.sum(q_row*q_row, axis=0) + 1e-6)
        state = state * decay[:, None]
        kv = tl.sum(k_row[:, None] * state, axis=0)           # erase 项：归约 K（D1）
        res = (v_row - kv) * bt
        state = state + k_row[:, None] * res[None, :]
        out_row = tl.sum(q_row[:, None] * state, axis=0) * scale
        tl.store(o_base + t*so_t + v_offs, out_row.to(out_ptr.dtype.element_ty), mask=v_mask)
    if OUTPUT_FINAL_STATE:
        # 终态 2D store：行连续取向避免标量化；钳位只钳行索引
        tl.store(sf_ptr + ..., state, mask=kv_mask)  # 或 STRIDED 分支 tl.trans(state)
```

**host 启动关键**（25x 与 0.135x 的唯一实质差异）：
```python
_cgdr_fwd_kernel[grid](..., multibuffer=True, unit_flag=True)   # ★ 25.57x 靠这个
```

### §4.4 ChunkedGatedDeltaRule 性能基准

| 项 | 值 |
|------|-------|
| 早期（基准污染） | **0.1351x**，framework 0.0379ms（torch 快速路径） |
| **当前（基准修正）** | **25.5704x** / impl 69.8us / framework 1.322ms（NPU2 复验 23.2953x） |
| 精度 | **24/50**（manual_iter，未达标）→ 早期 opt_iter_* 曾 50/50 |
| 关键改动 | `multibuffer=True, unit_flag=True` + STRIDED 加载行连续取向 |
| chunk 并行 | **未落地**（仅透传参数）；是未验证的潜力方向 |

**关键结论**:
1. **速度已达标（25.57x），精度未达标（24/50）** —— 接手的首要任务是修 24→50（早期 opt_iter_* 同 kernel 曾 50/50，差异在 verify 环境/阈值，需逐 case 复现），**不是**继续提 chunk 并行。
2. **`multibuffer=True, unit_flag=True` 是 Triton-Ascend 编译选项中的强力杠杆**：同一 kernel 加了它 impl 从 1083us → 69.8us（15.5x），配合基准修正共同造就 25.57x。首次生成 GDN kernel 时**默认就带**这两个选项，不必等到优化阶段。
3. **0.135x 教训**：看到 <1x speedup 先查 `framework.avg_latency_ms` 是否被污染（91 的 0.0379ms vs 1.322ms 是 35x 异常），别急着推翻结构。

---

## §5 FusedGdnGatingKernel 算子（gdn-gating）

**算子类别**: `gated-delta-net`
**典型特征**: GDN 门控片段（无递归）：输入 `a[M,D]`、`b[M,D]`、`A_log[D]`、`dt_bias[D]`，输出 `gate[M,D] = −exp(A_log)·softplus(a+dt_bias, beta)` 与 `beta_out[M,D] = sigmoid(b)`。纯 elementwise + 广播。
**性能基准**: 50/50 pass，几何平均加速比 **0.9738x** vs torch（**略低于 1x，待调优**）

### §5.1 Layer 1: 设计约束（FusedGdnGatingKernel 专属）

#### L1.1 softplus 必须 Taylor + threshold 双分支（数值 + 速度）
- **必须** `y = exp(BETA·x)`；`sp = where(y<0.2, _taylor_log1p(y), log1p(y))`，`sp *= 1/BETA`；再 `sp = where(BETA·x <= THRESHOLD, sp, x)` 兜底大值。
- **禁止** 裸 `log1p(exp(BETA·x))`（exp 溢出 → inf；小值区慢）。
- **Why:** D4。`BETA`/`THRESHOLD` 作 `tl.constexpr`；`_taylor_log1p` 是 8 项多项式（y−y²/2+y³/3−y⁴/4+…−y⁸/8），小值区比 log1p 快且够精度。

#### L1.2 按核分块：`rows_per_core` 二次分块
- **必须** `row_start = pid·rows_per_core`，`row_end = min(row_start+rows_per_core, M)`，内层再 `for row_base in range(row_start, row_end, ROWS_PER_BLOCK)` 分块处理 `ROWS_PER_BLOCK × BLOCK_D` tile。
- **Why:** 纯 elementwise 算子并行度来自 M（大），但 grid 不能超过核数；rows_per_core 让每个 program 摊多行，避免小 shape 时核空转（tensor-transform G4 的 GDN 应用）。
- **How to apply:** `BLOCK_D` 覆盖全 D（`offs_d = arange(0, BLOCK_D)` + mask）；A_log/dt_bias 是 [D] 广播，**预加载一次**出循环（循环不变量外提）。

### §5.2 Layer 2: 算法骨架

```python
@triton.jit
def fused_gdn_gating_kernel(A_log_ptr, a_ptr, b_ptr, dt_bias_ptr, gate_ptr, beta_out_ptr,
                            M, D, ..., rows_per_core,
                            BETA: tl.constexpr, THRESHOLD: tl.constexpr,
                            BLOCK_D: tl.constexpr, ROWS_PER_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * rows_per_core
    row_end = tl.minimum(row_start + rows_per_core, M)
    offs_d = tl.arange(0, BLOCK_D); mask_d = offs_d < D
    A_log = tl.load(A_log_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)   # 循环不变量外提
    dt_b = tl.load(dt_bias_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    A_log_exp = tl.exp(A_log); inv_beta = 1.0 / BETA
    for row_base in range(row_start, row_end, ROWS_PER_BLOCK):
        row_offs = row_base + tl.arange(0, ROWS_PER_BLOCK)
        mask_2d = (row_offs < M)[:, None] & mask_d[None, :]
        x = tl.load(a_ptr + ...).to(tl.float32) + dt_b[None, :]
        t = BETA * x
        y = tl.exp(t)
        sp = tl.where(y < 0.2, _taylor_log1p(y), tl.log1p(y)) * inv_beta
        sp = tl.where(t <= THRESHOLD, sp, x)                   # 大值兜底防 inf
        gate = -A_log_exp[None, :] * sp                        # gate = −exp(A_log)·softplus
        beta_out = 1.0 / (1.0 + tl.exp(-b_tile))               # beta = sigmoid(b)
        tl.store(gate_ptr + ..., gate.to(gate_ptr.dtype.element_ty), mask=mask_2d)
        tl.store(beta_out_ptr + ..., beta_out.to(...), mask=mask_2d)
```

### §5.3 Layer 3: 关键技巧

- **`_taylor_log1p` 8 项多项式**：`y−y²/2+y³/3−y⁴/4+y⁵/5−y⁶/6+y⁷/7−y⁸/8`，全部用常量系数（`0.25`、`1.0/3.0` 直接写死），避免编译期除法。
- **循环不变量外提**：`A_log_exp`、`inv_beta`、`dt_b` 在 `row_base` 循环外算一次（D 维广播）。
- **与 91 的 stable softplus 对照**：91 的 `where(gi>20, gi, log(1+exp(gi)))` 是简化单分支（无 beta 缩放）；82 是带 `BETA`/`THRESHOLD` 的通用版。首次生成按算子需求选分支数。

### §5.4 FusedGdnGatingKernel 性能基准

| 项 | 值 |
|------|-------|
| 全量 | **0.9738x** / 50/50 |
| 结构 | 单 kernel 纯 elementwise，rows_per_core 分块 |
| 优化空间 | 未做 BLOCK_D/ROWS_PER_BLOCK 扫描；0.97x 接近 1，属"正确但未提速"样本 |

**关键结论**: 纯 elementwise 门控 kernel 的正确性模板（Taylor + threshold + 广播外提）可直接复用；性能 0.97x 说明该类算子瓶颈在 launch/内存带宽而非计算，调参空间有限（tensor-transform G4 的核数控制适用）。

---

## §6 常见陷阱与避免方法

### §6.1 OlmoHybridGdnFullForward 陷阱

| 陷阱 | 现象 | 正确做法 |
|------|------|---------|
| recurrence fp16 累加 | verify 27/50 | D3：状态全程 fp32，只写 output 时转 dtype |
| 固定 bv=128 无 dim 保护 | 族B（HVD=64）编译崩 grid=0 | D6/L1.2：`bv = min(bv, head_v_dim)` |
| kernel 内残留 Python 串行循环 | 初始版 4/50、~0.9x | 全 kernel 化，`for t` 在 Triton kernel 内 |
| 输出投影低精度 dot | 精度超标 | L1.3：fp32 dot 对齐 `F.linear(x, W.float())` |
| 试图跨 T 并行 recurrence | 不可能 | D2：串行链，结构性硬上限，tile 只能减常数 |

### §6.2 Gdn2Fwd 陷阱

| 陷阱 | 现象 | 正确做法 |
|------|------|---------|
| erase 项顺序错（先加外积再 erase） | 结果错 | L1.2：先 `erase=sum(bk·S)` 再 `v_new=w·v−erase` 再外积 |
| varlen 段内 T 偏移写死 | 段间状态串扰 | L1.3：逐段 `for n` 启动，T_start/T_end 传段界 |
| QK norm 用 `k/sqrt(sum)` | 除零/溢出 | D4：`rsqrt(sum+EPS)` |
| V_TILE 固定 64 不按 K_PAD 面积调 | 小 K 时并行度浪费 | D5：V_TILE 按 `K_PAD×V_TILE≤1024` 放宽 |

### §6.3 ChunkedGatedDeltaRule 陷阱

| 陷阱 | 现象 | 正确做法 |
|------|------|---------|
| **看到 <1x speedup 直接归咎结构** | 0.135x 实为基准污染（torch 0.0379ms） | **L1.1：先查 `framework.avg_latency_ms`，与复验跑对比，再决定是否动结构** |
| 忘加 `multibuffer/unit_flag` | kernel 1083us，错失 15.5x | §4.3 host 启动必须带；首次生成默认加 |
| STRIDED_STATE 全偏移钳位 | gather 标量化 / 507035 | L1.4：`[V,K]` 行连续 + 只钳行索引 |
| 终态 2D store tile 面积超限 | AICore 异常 / 性能崩 | L1.5：`BLOCK_V = 1024 // BLOCK_K` |
| 运行时 bool 分支杀性能 | 变体多时 kernel 膨胀 | L1.3：全部 `tl.constexpr` |
| softplus 裸 `log(1+exp)` | exp 溢出 | D4：`where(gi>20, gi, log(1+exp(gi)))` |
| QK norm 用 `sqrt/maximum` | 慢 + 除零边界 | D4：`rsqrt(sum+EPS)` |

### §6.4 FusedGdnGatingKernel 陷阱

| 陷阱 | 现象 | 正确做法 |
|------|------|---------|
| 裸 `log1p(exp(BETA·x))` | exp 溢出 inf | L1.1：Taylor + threshold 双分支 |
| A_log/dt_bias 在行循环内重复 load | 冗余 MTE | L1.2：循环不变量外提 |
| grid 超过核数空转 | 小 shape 慢 | rows_per_core 二次分块（tensor-transform G4） |

---

## 附：四算子共同公式速查

| 阶段 | 公式 | 算子 |
|------|------|------|
| 门控衰减 | `g_t = exp(gate_t)`；68 门控 `g = sigmoid(gate)·exp(dt_bias·A_log)` | 68/82/91/94 |
| 稳定 softplus | `sp = where(y<0.2, taylor8(y), log1p(y))`；91 `where(gi>20, gi, log1p(exp(gi)))` | 82/91 |
| QK L2 norm | `k ← k · rsqrt(Σk² + EPS)`；68 用 running-stats `k·rms` | 68/91/94 |
| 状态递归 | `S_t = decay·S_{t-1} + k_t·v_t^T`（68 无 erase）；`S_t = decay·S_{t-1} + k_t·(w·v − b_t·S)^T`（91/94 erase） | 68/91/94 |
| 输出 | `o_t = scale·q_t^T·S_t`；68 额外 gated RMSNorm + `out@W_o^T` | 68/91/94 |
| 终态 | 可选写回 `final_state = S_T` | 91/94 |
