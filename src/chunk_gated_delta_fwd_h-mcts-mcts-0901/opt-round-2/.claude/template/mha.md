---
name: mha
description: MHA 家族（一 标准类·含投影段：4 个 nn.Linear 投影 + softmax(QKᵀ)@V 的 MHA / SelfAttention / CrossAttention）的 Triton Ascend 优化经验合集，含投影段终态架构、收益归因与证伪方向；attention 主链约束与 FA 卡交叉使用
metadata:
  type: reference
---

# 一·标准类 · 含投影段（MHA / SelfAttention / CrossAttention）算子优化经验

本文件按 `level4写法分类体系.md` 的写法口径，负责 **一 标准类 · 含投影段变体**：
参考实现含 **4 个 `nn.Linear(D,D)` 投影**（Q/K/V 投影 + 输出投影），attention 主链是
三段式或分块流式 `softmax(Q@Kᵀ/√d)@V`。该细分下的算子是
**17 MultiHeadAttention · 18 SelfAttention · 19 CrossAttention**（self/cross 只是输入来源不同，
写法同构）。

- **§1 判别特征与分类依据**（为什么独立成卡，附实测差异）
- **§2 与 `flash_attention.md` 的交叉使用**（投影段硬约束索引，两份 Layer 1 都摘录）
- **§2.5 本卡 Layer 1 补充硬约束**（M1~M4：NUM_CORES 动态读取 / grid 恒定 / UB 前置剔除 / dot 链 ≤2）
- **§3 实证：SelfAttention**（13 轮，1.1154 → 1.797 旧口径 / 2.4690 新口径，三 kernel 终态架构）
- **§4 实证：CrossAttention**（18 轮 opt_iter_0..17，1.8208 → 2.1125，含 K/V 投影融合）
- **§5 实证：MultiHeadAttention**（17 轮 opt_iter_0..17，0.1531 → 2.0439，13.35x，split-K + ieee dot 链路）

> **证据基础**：三个算子各 50 case 的完整优化轨迹（fp32/fp16/bf16 混合，
> `head_dim ∈ {20..80}` 普遍非 2 幂，S ≤ 897），910_93 / CANN 9.1.0 / triton-ascend 3.2.2。
> 三者精度均 **50/50**。

---

## §1 判别特征与分类依据

### §1.1 判别特征（决定用不用本文件）

打开参考实现，**同时满足**：

1. 含 **4 个 `nn.Linear(D,D)` 投影**（Q/K/V 各一 + 输出投影；self 形态 `q=k=v=x`，
   cross 形态 `query` 与 `context` 分开输入、`key=value=context`）；
2. attention 主链是 `softmax(Q@Kᵀ/√d)@V`（三段式或 KV 分块流式均可）；
3. 投影权重常由参考实现内部 `manual_seed` 现场初始化，`head_dim = D/n_heads` 普遍非 2 的幂。

### §1.2 为什么不能落到已有行（实测差异，按索引"新增子类的规矩"）

- **与一·1 基础三段式 / 一·2 分块流式（行 7/11，`flash_attention.md`）的差异**：
  本细分多了**投影段**这个独立优化对象，且实测**最大的一笔收益在投影侧，不在 attention 侧**——
  CrossAttention 权重预转置单笔 -15%（bf16 +25%），GQA 含投影形态投影侧 +57%（见 `gqa.md` L1.6）。
  FA 卡的注意力侧经验（online softmax、BLOCK 面积预算）只覆盖一半目标函数。
- **误分类代价有实测记录**（§4.5）：CrossAttention 曾被归入 `sdpa` 行，
  投影权重以 `[out,in]` 原始布局进 GEMM，Phase 4 花 17 轮才 rediscover 预转置。
  **首次生成不命中本卡 = 丢掉最大的一笔收益。**

## §2 与 `flash_attention.md` 的交叉使用（⚠️ 交叉行，两份 Layer 1 都摘录）

本细分是**投影段（本卡）＋ attention 主链（FA 卡）**两段结构，优化对象互相独立
（profiling 里分别看 Duration 占比，主攻目标每轮会换，实测从 attention 64%/linear 36%
变到 46%/54%）。`precheck.json` 的 `layer1_constraints_loaded` 必须**同时摘录两份**：

| 段 | 必读条目（`flash_attention.md`） |
|---|---|
| 投影段 | **L1.11**（host 侧先对齐 nn.Linear 语义，权重错与转置错互相掩盖）、**L1.12**（投影权重禁止 `[out,in]` 原始布局喂 GEMM，★★★ 头号杠杆）、§3.4（两段结构）、§4.4（权重按 tile 预重排）、§4.5（投影分块 dtype 分档 + 并行度保护） |
| attention 主链 | L1.1（fp32 ieee dot）/ L1.3（online softmax 同步缩放）/ L1.4（精度契约）/ L1.13（BLOCK 面积预算）/ §4.6（half 路径 PV hi/lo 双 dot） |

另需摘录本卡 **§2.5 的 M1~M4 四条补充硬约束**（原拟写入 FA 卡，经 review 权衡收敛到本卡，仅对本细分生效）。

⚠️ **不要**把 4 次投影和 1 次 attention 合并进同一个 `@triton.jit`（`MODE: tl.constexpr` 分支）——
见 `flash_attention.md` §7.2。

### §2.5 本卡 Layer 1 补充硬约束（M1~M4，与 FA 卡 Layer 1 一同摘录）

> 这四条在 MHA 家族任务中发现，原拟写入 `flash_attention.md` 作为通用约束；
> 经 MR review 权衡**收敛到本卡**——避免对纯 FA 算子引入未在该路径验证过的硬边界。
> `precheck.json` 的 `layer1_constraints_loaded` 必须逐条摘录。

- **M1 ★ `NUM_CORES` 禁止硬编码 24，模块导入时按设备动态读取一次**
  ```python
  # host（模块导入时动态读取一次，进程内仍是常量）
  try:
      import triton.runtime.driver as driver
      _dev = torch.npu.current_device()
      NUM_CORES = driver.active.utils.get_device_properties(_dev)["num_aicore"]  # 910_9372 实测 20
  except Exception:
      NUM_CORES = 20
  ```
  `NUM_CORES` 必须保持**进程内常量**——做成随 case/随 launch 变化的值会触发编译爆炸
  （实测 verify 超时、0 个用例启动）。硬编码 24（DAV 口径）在 910_9372（实际 20 核）的
  persistent 步长循环下失配 4/24，即尾部排队 + 负载不均（17_MHA 实测修正后 **+11.6%**）。
  `grid` 是启动参数，随 case 变化无妨。

- **M2 ★ BLOCK 由 tuner 注入时，`grid=(NUM_CORES,)` 恒定 + 核内 grid-stride**
  host 在 launch 时**无法预知 tuner 选中 config 的 tile 总数**，按旧 config 的 tile 数裁剪
  grid 会在 tuner 换 config 后产生 stride 空洞、**静默漏 tile**。
  正解：`grid = (NUM_CORES,)` 恒定，kernel 内 `for t in range(pid, TOTAL, NUM_CORES)`。

- **M3 ★ UB 溢出 config 必须在 `early_config_prune` 里按峰值估算前置剔除**
  `_bench` 只捕获 OutOfResources 等三类异常，UB 溢出的通用异常会**炸掉整个 autotune 进程**。
  峰值估算公式见 §3.1 K3 / §4.1 K1（CAP 168~200KB 按算子实测标定）。

- **M4 ★ 单 kernel 循环体内同 dtype cube `tl.dot` 链 ≤ 2 个（aicore 507014）**
  禁止同一循环体放 3 个 `tl.dot`（如 fp32 的 hi@hi+hi@lo+lo@hi 三 dot 补偿）——M>256 时
  aicore timeout（error 507014）必挂；join/reshape 堆叠成单大 dot 或 K 维堆叠 2 dots 会让
  bishengir-compile 直接崩溃。ieee 路径 QK 1 dot + PV 1 dot、half PV hi/lo（FA 卡 §4.6）
  均为 2 dot/iter，天然合规。

---

## §3 实证：SelfAttention（`mha-self`）（13 轮 opt_iter_0..12，1.1154 → 1.797 旧口径 / 2.4690 新口径）

> 形态：`q=k=v=x`（self-attention），4 个 `nn.Linear(D,D,bias=False)` 投影
> （`manual_seed(42)` 现场初始化），`softmax(Q@Kᵀ/√d)@V`；50 case = fp32 17 / fp16 17 / bf16 16，
> `head_dim ∈ {20..80}` 普遍非 2 幂，S ≤ 897。架构 910_93 / CANN 9.1.0 / triton-ascend 3.2.2。
> 本章是**终态架构 + 判据 + 证伪**，目标：同类算子首次生成直接命中，Phase 4 少走弯路。
>
> ⚠️ **诚实口径**：最优为 opt_iter_9（优化点 17：删 K3 冗余 p 二次 where，fp32 下溢为精确 0、
> 逐比特等价），双跑 per-case 均值 **1.797**（run1=1.8275 / run2=1.7616），vs Phase 3 基线
> 1.611x。config target 0.8 达成，**用户口述目标 2.0 未达**：分 dtype fp32 1.49 / fp16 1.91 /
> bf16 2.00（仅 bf16 破 2.0）。标题中的 2.4690 是 benchmark 口径修正（多 launch 少计，
> 新仓 6e4dc13c）后**同一份代码**的重测值——纯测量修正，跨版本数字不可比（§3.2-④）。
> **剩余瓶颈**：worst6 中 5/6 为 K1 投影 GEMM 主导（66-77%），fp32 大 shape K1 40.6 TFLOPS
> vs torch 57 TFLOPS（Triton cube 调度差距）；msprof 采集证伪了"cube 物理极限"结论
> （K1 load-bound MTE2 33%、K3 vector-bound 44-60%，MMAD 均未进前三）。
> 定向三连拒（双跑判定）：K1 追加 (64,256,64) 配置 / K3 v-load 提前 / K1 grouped ordering。

### §3.1 终态架构（三 kernel 结构，逐项都有实测依据）

```
K1  _linear_kernel（投影 GEMM，每 forward 发 2 次）
  · MERGE_QKV：q=k=v=x ⇒ W_qkv = cat([Wq,Wk,Wv], 0)，单次 GEMM 与三次独立投影【逐比特一致】
  · W host 侧预转置 [K,N]（= FA 卡 L1.12，IR 实证消除 k-loop 内 tl.trans→vtranspose）
  · fp32 走 input_precision="ieee"（= FA 卡 L1.1）；fp16/bf16 原生 dtype 单 dot
  · BLOCK_M/N/K ∈ {64,128}×{64,128}×{64,128} 的 6 配置 autotune（key=['M','N','K']）
  · Cube 类 kernel 必须手写 Config（hints/自动模式仅支持 Vector）
  · grid=(NUM_CORES,) 恒定 + 核内 grid-stride（BLOCK 是 tuner 注入的，见 §2.5 M1/M2）

K3  _attn_kernel（online softmax）
  · QKV 物理布局 [B*S, 3D]：q 列 h*HD、k 列 D+h*HD、v 列 2D+h*HD —— strided 逻辑视图免 transpose（= FA 卡 L1.6）
  · fp32 路径：q/k/v 直载，QKᵀ 与 P@V 各 1 个 ieee dot（每 k-iter 6 dot → 2 dot）
  · half 路径：p kernel 内 hi/lo 分裂 2-dot 补偿（= FA 卡 §4.6；p 直接降 dtype 的误差 5e-4/4e-3 顶穿阈值）
  · BLOCK_SQ/BLOCK_SK ∈ {32,64,128}² autotune + early_config_prune UB 剪枝：
      ub = BSQ*BD*esz + 2*BSK*BD*esz + BSQ*BD*4 + 2*BSQ*BSK*4 [+ half 路径 2*BSQ*BSK*esz]
      CAP = 168KB（实测 (64,64,128) fp32 可编译 = 160KB，留 8KB 余量）
    ⚠️ UB 溢出的通用异常不在 _bench 捕获的三类里，会炸掉整个 autotune 进程 ⇒ 必须前置剔除
  · scale 保留 tile 内乘（q 预缩放已证伪，见 FA 卡 §5.2）；末尾 out = acc * (1.0/l) 广播乘

host
  · 权重复刻：rng save/restore + manual_seed(42) + CPU fp32 uniform_(-1/√D) ×4 顺序 q,k,v,o
    （= FA 卡 L1.11；AST 禁 nn.Linear 时用数值等价初始化复刻）
  · 缓存 key=(D, H, device, dtype)，权重只建一次
```

### §3.2 与 FA 卡（`flash_attention.md`）既有条目的冲突判据（4 条，搬运结论前先核对）

| # | 冲突 | SelfAttention 实测 | **判据** |
|---|------|--------------------|----------|
| ① | F3 `ceil16` | 打平（-0.15%）拒绝 | `ceil16` 省的是 K3 tile 面积；**K1 GEMM 主导的算子无收益**。先看 per-kernel 时延占比再决定投入 |
| ② | L1.14 `propagate_nan` 分档 | 无真实效果拒绝（会话间翻号） | `m_new` 更新占 K3 比例小 / K1 主导时低优先级；FA 卡的 +2.7% 来自 attention 主导的算子 |
| ③ | §4.1 末尾 `acc/l` 倒数乘 | 我们采纳了（`1.0/l` 标量除 + 广播乘） | 与「循环外无收益」的失效边界记录冲突，case 相关；成本极低可顺手做，判定按打平处理 |
| ④ | §7.2 benchmark 口径 | 新仓 6e4dc13c 已修正多 launch 少计 | 同一份代码旧口径 1.7970 → 新口径 2.4690，**纯测量修正**；跨版本数字不可比，报告必须注明 commit |

### §3.3 判定方法论（噪声地板，比优化本身更早决定成败）

- **单 case 噪声带 ±30%**：HD=64、代码路径逐比特相同的 case 单跑波动 -31% ⇒ **单 case 对比不可信，
  per-case 均值 geomean 是唯一稳定口径**（双跑取每 case 均值再求几何平均）。
- **跨会话系统漂移 ±2%**：逐比特相同代码的 27 个 case，跨会话对比均值 -2.40%，同会话背靠背对拍翻为 +1.27%。
  ⇒ **<±2% 的 delta 跨会话对比不可信；关键判定必须同会话背靠背（base→cand→base）对拍**。
- **微优化真实效应 <1%**（分档 maximum、ceil16、指令重排级别）低于噪声地板 ⇒ 这类改动只在
  「有 IR 级证据（指令条数变化）+ profiling 证明落在关键路径」时才值得一跑。
- 配置扩展的教训（iter10）：**大 shape 微基准增益不能线性外推全量 geomean**——tuner 在小 case
  上被 do_bench 噪声误选的概率随配置空间增大而上升。

### §3.4 收益归因（本算子 13 轮的排序，指导同类算子的投入顺序）

| 排序 | 改动 | 收益 |
|------|------|------|
| 1 | fp32 ieee dot 替换 hi/lo 分裂链（微基准 3.8x，17_MHA 链路 +13.1%） | 数量级 |
| 2 | W 预转置 [K,N] + MERGE_QKV 单 GEMM（L1.12 + 融合） | 大 |
| 3 | BLOCK autotune + UB prune（L1.13 思路）+ grid-stride 解耦 | 中 |
| 4 | 冗余边界运算消除（p 的二次 where、q 预缩放证伪后的保守保留） | 小但零风险 |
| 5 | ceil16 / propagate_nan / 指令重排 / 配置扩展 | 0（全部拒绝） |

### §3.5 ⛔ 证伪方向（双跑 / 全量判定，不要重跑）

| 死路 | 实测 |
|------|------|
| autotune 配置扩展（补 BLOCK_K=128 / 大组合，iter10） | **拒绝**：大 shape 微基准增益（-15%/-26% 时延）被小 shape 的 do_bench 噪声**误选**对冲，全量 geomean 打平偏差。配置空间一大，tuner 在小 case 上选错的概率也随之上升 |
| k/v load 提前到 QKᵀ dot 之前发射（iter11） | **拒绝**：提前量与 scores 计算在编译器调度下对冲，geomean 无净收益 |
| tile 按 M 分组重排发射顺序（grouped ordering，iter12） | **拒绝**：大 shape +8~23% 与小 shape -9~18% **极化对冲**；受益/回退的 M 区间重叠，无干净门控分界面，门控预期仅 +0.3% |
| `ceil16(HD)` 替代 `next_pow2(HD)`（iter13） | **打平拒绝**（-0.15%）：HD∈{72,78,80} 的 case 瓶颈在 K1 GEMM，K3 的 BLOCK_D 面积收益对冲。见 §3.2-① |
| `propagate_nan=ALL` 分档（iter14） | **无真实效果拒绝**：两次独立会话对拍 gated 组 +1.63% → -4.12% 翻号，属跨会话噪声。见 §3.2-② |

---

## §4 实证：CrossAttention（`mha-cross`）（18 轮 opt_iter_0..17，1.8208 → 2.1125）

> 形态：`query [B,S_Q,D]` 与 `context [B,S_K,D]` 分开输入，`key=value=context`，
> 4 个 `nn.Linear(D,D,bias=False)` 投影（`manual_seed(42)` 现场初始化），
> `softmax(Q@Kᵀ/√d)@V`；50 case = bf16 17 / fp16 16 / fp32 17，`head_dim ∈ {20..80}`
> 普遍非 2 幂，S_Q ≤ 897。架构 910_93 / NPU=6 / CANN 9.1.0 / triton-ascend 3.2.2。
> 目标：同类算子首次生成直接按 §4.1 出架构，Phase 4 只验证不探索。
>
> ✅ **复用验证（0815 重生成）**：凭本卡经验重跑同一算子，Phase 3 **首版即 2.4123**
> （权重预转置 / ceil16 / fp32 ieee dot 首轮全部生效），仅 opt_iter_1 一轮 K/V 融合微调
> （impl_ms -4.0%）收敛至 **2.4391**——本卡"首次生成直接命中"的目标已被实测兑现。

### §4.1 ★★★ 终态架构（三 kernel，首次生成直接采用）

```
K1  _ca_linear_kernel（通用投影 GEMM，每 forward 发 2 次：Q 投影 + O 投影）
  · W host 预转置 [out,in]→[in,out]（= FA 卡 L1.12，头号杠杆，首次生成必须做对）
      kernel 内 w_off = offs_k[:,None]*D + offs_n[None,:]（BN 方向 stride=1）
  · fp32 走 input_precision="ieee"（= FA 卡 L1.1）；fp16/bf16 原生 dtype 单 dot
  · BM/BK/BN ∈ {64,128}×{32,64}×{64,128} 7 配置 autotune + early_config_prune
      ub = 2*(BM*BK + BK*BN)*esz + BM*BN*4 < 200KB
  · grid = min(num_cores, tiles) + 核内 grid-stride（= §2.5 M1/M2）

K2  _ca_kv_linear_kernel（★ K/V 投影融合，1 次启动替代 2 次）
  · key=value=context ⇒ K=ctx@Wk 与 V=ctx@Wv 共享同一输入张量
    ⇒ 单 kernel 内双 dot 共享 x load，省 1 次 context memory pass + 1 次 launch
  · 循环体内：load x 一次 → wk/wv 两次 load → acc_k/acc_v 两个 fp32 累加器双 dot
  · UB 占用近乎翻倍（x + 2×w + 2×acc）⇒ autotune 配置数裁到 4，防 verify 超时
      ub = (BM*BK + 2*BK*BN)*esz + 2*BM*BN*4 < 200KB
  · 单独实测紧邻 A/B impl ratio ≈ 0.96（真实 ~4%），与权重预转置叠加后合力 -15%

K3  _ca_attn_kernel（online softmax，S_Q ≠ S_K）
  · K 直接 load 成 [BLOCK_D, BSK] 转置态，省 tl.trans（FA 2.12，load 路径收益）：
      kt_col_base = b*S_K*D + h*HD + offs_d[:,None]
      kt_off = kt_col_base + offs_k[None,:]*D      # d 方向连续
  · fp32：QKᵀ 与 P@V 各 1 个 ieee dot；fp16/bf16：QKᵀ 原生 dot，PV hi/lo 双 dot（= FA 卡 §4.6）
  · BLOCK_D = ceil16(HD)；BSQ/BSK ∈ {32,64,128}² autotune + UB prune（CAP 192KB）
  · p 掩码必须显式归零 p = tl.where(sk_valid, p, 0.0)
    （A1：Ascend 硬件 exp(-3e38) 残留 ~1e-16 非零，删二次 where 在 bf16 小 shape 上 MERE 违规）
  · scale 保留 tile 内乘 scores*scale_inv（q 预缩放已证伪，FA 卡 §5.2）

host
  · 权重复刻：rng save/restore + manual_seed(42) + CPU fp32 uniform_(-bound,bound) ×4
    顺序 q,k,v,o；bound = √3·√(2/(1+a²))/√D（Kaiming linear 公式逐字复刻 = FA 卡 L1.11）
  · ★ 预转置放 host：w.t().contiguous() 不消耗 RNG，数值逐位不变，参考实现不受影响
  · 缓存 key=(D,H,device,dtype)，权重只建一次；cache.clear() 防多 dtype 累积
```

### §4.2 收益归因（18 轮排序，全部经紧邻 A/B impl_ms 判定）

| 排序 | 改动 | 紧邻 A/B impl_ms ratio | 结论 |
|------|------|----------------------|------|
| 1 | 权重预转置 [out,in]→[in,out]（L1.12，4 个投影全改） | 0.85（真实 -15%） | **主收益**，分 dtype bf16 +25%/fp16 +18%/fp32 +5.5% |
| 2 | K/V 投影融合（K2 kernel） | ≈0.96 | 真实小幅，与 1 叠加 |
| 3 | K 转置直读（FA 2.12） | 组合内难剥离 | 文档值 +0.7%，零风险顺手做 |
| 4 | 循环不变量外提 / V load 延迟 / ceil16 / qd 外提 | 1.00~1.01 | 全部噪声内，可做可不做 |

终态：geomean 1.8208 → 2.1125（+16.0% impl_ms），50/50 verify，bf16 2.3755 / fp16 2.2306 / fp32 1.7849。

### §4.3 ⛔ 证伪方向（不要重跑）

| 死路 | 实测 |
|------|------|
| `multibuffer=True` | UB 紧张时**全 shape 劣化**（约 -10%），与 FA 卡 §5.2 记录一致 |
| K/V host 侧 `torch.cat` 后单 kernel | 每 call 拼接开销吃掉融合收益——**融合必须在 kernel 内做，不在 host 做** |
| ceil16 替代 next_pow2（单独提交） | HD∈{72,78,80} 个别 case +4~10%，HD≤32 回退，geomean 净 ≈ 0（1.011）。首次生成直接用 ceil16 即可，但**不要指望它单独带来收益** |
| 删除 p 的二次 `tl.where` 归零 | cross_attention bf16 [2,59,360] MERE 违规（49/50），见 exp_debug_008；SelfAttention 上却 50/50——**场景条件性，默认保留归零** |
| qd（同 dtype cast）循环不变量外提 | no-op cast，0 收益 |

### §4.4 ★ 测量口径（与 FA 卡 §7 互补：`speedup_vs_torch` 会被框架噪声污染）

- **污染机理**：speedup = framework_ms / impl_ms。NPU 繁忙时 framework（torch 参考）膨胀
  幅度 > impl 膨胀幅度，speedup 被动抬高。**同一份代码两次 benchmark 可在 1.79~2.22 之间波动**。
- **正确判定**：紧邻 A/B —— baseline 与 optimized 在同一脚本内背靠背（base→opt→opt→base），
  repeats≥30，读 `per_shape_results[].implementation.avg_latency_ms` 算 geomean(opt/base)。
  ratio < 0.95 才算真加速；0.95~1.05 视为噪声，不得写入报告当"优化成功"。
- **报数纪律**：report/summary 中每个数字必须注明口径（紧邻 A/B impl_ms 还是单跑 speedup_vs_torch）。
  用户会问"有没有作弊"——宁可报真实的 1.8x 也不报虚高的 2.2x。

### §4.5 分类教训（为什么会多走 10+ 轮）

本次 Phase 2 曾把 CrossAttention 归入 `attention.md`（旧分类表把 cross-attention 列在
`sdpa` 行），导致 L1.12 / §4.6 / ceil16 这些 FA 卡的硬规则**首次生成时没有生效**，
权重以 `[out,in]` 原始布局进了 GEMM，Phase 4 花了 17 轮才从零 rediscover。
**判别口诀：只要有"4 个 Linear 投影 + softmax(QKᵀ)@V"结构，无论 self/cross/MHA，
一律走本文件 + `flash_attention.md` 交叉行**（`attention_index.md` 判别表 6a 行）。

---

## §5 实证：MultiHeadAttention（`mha`）（17 轮 opt_iter_0..17，0.1531 → 2.0439）

> 形态：`query [B,S_Q,D]` / `key,value [B,S_K,D]` 分开输入，`n_heads` 为 attr，
> 4 个 `nn.Linear(D,D,bias=False)` 投影（`manual_seed(42)` 现场初始化）；50 case =
> fp32 17 / fp16 17 / bf16 16，`head_dim ∈ {20..80}` 非 2 幂，`S_Q ≤ 897 / S_K ≤ 907`。
> 架构 910_93 / CANN 9.1.0。终态 **2.0439**（三次运行 2.0227/2.0465/2.0439 取中位数），
> 50/50，分 dtype fp32 1.7953 / fp16 2.1285 / bf16 2.2469，达 target 2.0。
> vs Phase 3 基线 **13.35x**——本卡三个算子里基线最低、提升空间最大的一例。

### §5.1 终态架构（四 kernel，比 §3/§4 多 split-K 一路）

```
K1  _mha_linear_kernel（投影 GEMM ×4，autotune BLOCK_M/N/K + UB prune）
K2  _mha_attn_kernel（online softmax 主路，autotune BLOCK_SQ/SK + UB prune）
K3  _mha_attn_partial_kernel（★ split-K 部分态：小 grid 欠载时 S_K 分裂）
K4  _mha_attn_combine_kernel（LSE 合并 partial 结果）
  · split-K 门控：tiles*2 ≤ cores 且 nkb ≥ 4 才启用（opt_iter_14 收紧后采纳）
  · 其余同 §4.1：W 预转置 / fp32 ieee dot / half PV hi/lo / ceil16(HD) / UB prune autotune
```

### §5.2 收益归因（17 轮排序，chip 8 口径）

| 排序 | 改动 | 轮次 / 效果 |
|------|------|------------|
| 1 | **fp32 原生 ieee cube dot 全链路**（`input_precision="ieee"`，推翻"fp32 禁 tl.dot"） | opt_iter_16，**+13.1%**（1.6289→1.8419）；attn 每 k-iter 6 dot→2 dot，linear 微基准 3.8x，→ FA 卡 L1.1 |
| 2 | linear+attn **BLOCK autotune + early_config_prune UB 剪枝**（tile 计数下沉 kernel） | opt_iter_17，**+11.0%**（→2.0439 达标） |
| 3 | pass 合并 / hi-lo 分裂 dot / 生产者预分裂等批量改造 | opt_iter_0..7，0.1531→1.6106 |
| 4 | attn **split-K + LSE combine**（小 grid 欠载）+ 门控收紧 | opt_iter_13/14，+0.4% |
| 5 | IR 驱动 vdiv→倒数乘法（scores/epilogue） | opt_iter_15，+0.3% |

### §5.3 ⛔ 证伪方向（不要重跑）

| 死路 | 实测 |
|------|------|
| q/k/v 多指针融合成单次启动 | **-43%**（0.9086），坚决不做 |
| half 路径 PV 单 dot（省 hi/lo） | 精度证伪——PV 必须 hi/lo 双 dot（FA 卡 §4.6 同结论） |
| 独立预分裂 pass / k-loop 地址外提 | 打平，拒绝 |

### §5.4 与 §3/§4 的分工

- 17_MHA 是 **ieee dot（FA 卡 L1.1）与 dot 链 ≤2（本卡 §2.5 M4）两条硬约束的发现算子**；
- §3（self）特有 MERGE_QKV 单 GEMM（q=k=v=x 逐比特一致）；§4（cross）特有 K/V 投影融合
  （key=value=context 共享 x load）；§5（MHA）特有 split-K + LSE combine（S_Q≠S_K 且
  grid 欠载时的兜底）。三者投影段其余部分同构，首次生成按形态各取所需即可。
