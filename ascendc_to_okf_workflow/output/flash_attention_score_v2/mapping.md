# P2 语义转化决策表 — flash_attention_score（v2，服务 level4 SelfAttention）

> 依据：`output/flash_attention_score_v2/extraction.md`（锚点) + `ascendc_triton_semantic_mapping.md`（速查表）+ `backend_facts/backend_contracts.md`（后端契约，命中即引用）。
> **服务目标**：level4 SelfAttention 的 Triton 生成。golden 与 FAS 的接口差在本表 §0 显式拆解。

## 0. 目标算子与 FAS 范围差（先拆接口，再谈映射）

level4 SelfAttention（`level4_target/SelfAttention/definition.json` 内嵌 reference）：
```
q=q_proj(x).view(B,S,H,hd).transpose(1,2); k=…; v=…
scores = matmul(q.float(), k.float().T) / sqrt(hd)
weights = softmax(scores, dim=-1)
out     = matmul(weights, v.float()).to(dtype)
return out_proj(out.transpose(1,2).contiguous().view(B,S,D))
```
- **接口差**：golden 在注意力前**内建 4 段 `nn.Linear` 投影**（q/k/v/out_proj, seed=42），且 out 再经 `out_proj`。FAS CANN kernel **只覆盖注意力核** `softmax(QKᵀ·scale)·V`（extraction §1.1）。
- **→ P2 映射结论**：Triton 侧投影 = MatMul，归 **wrapper（或融合进同一 kernel 的前导 dot）**，二者与注意力核之间的 head 重排（`S→(H,hd)` + `transpose(1,2)` = 非尾轴 movedim+contiguous）在 wrapper 完成。level4 的 query_length==key_length==S，为**朴素自注意力**。
- **level4 用不到 FAS 的可选面**：pse/atten_mask/dropout/sparse/NZ/fp8/rope 全部不出现（definition.json 仅 x + n_heads）。mapping 主纲=**稠密 flash 注意力**；可选面标「可达但本目标不触发」，不阻塞主骨架。

## 1. 可达性判定表（P1 事实 → Triton 决策 → 三态 → 锚点）

| P1 事实（锚点） | Triton 决策 | 态 | 依据 |
|---|---|---|---|
| 常量公式 `softmax(QKᵀ·scale)·V`（extr §1.1） | flash 注意力主循环：`qk=dot(q,kᵀ)·scale → online softmax → acc=dot(p,v)` | ✅ | 映射表 §3 row42/44/50 |
| 4 段 nn.Linear 投影（definition.json reference） | wrapper 内 4×`tl.dot` 或 torch.matmul 完成；与注意力核串接 | ✅ | 速查表 row42；接口差 §0 |
| `scale=1/√head_dim`，`head_num` 属性（def.cpp:161, :157） | `head_dim = D//n_heads` wrapper 算出；`SCALE` 传 `tl.constexpr`（=1/sqrt(hd)） | ✅ | 速查表 row65（constexpr 两态；默认常量折叠利好） |
| 4 布局归一化 NSD/BSH/SBH/BNSD/TND（template_tiling_key.h Layout 位） | wrapper 读 `tensor.stride()` 归一化，kernel 内只用 stride 向量线性 offs；不入 kernel 分支 | ✅ | 速查表 row20/31；P1 §4.1.3 tilingkey |
| 多核任务轴 = 压平 (B·N2·G·S1)，S2 不进多核（s1s2_bn2gs1.h:773,861） | `task_blocks = B*H*cdiv(S,BLOCK_M)`（S1==S2==S，N2=H）；`grid=(task_blocks,)`；kernel 内 `for kv in range(0,S,BLOCK_N)` 内层 | ✅ | 速查表 row12/13 persistent-core |
| 核内连续段分核（persistent-core）（s1s2:773-778） | `grid=min(task_blocks, aic)`；`for blk in range(pid*tpb, …)` 连续段 | ✅ | 速查表 row13（禁交织、禁「一块一 program」） |
| C 侧 bmm1/bmm2 基本块 128×128 / K=64/128（tiling_general.cpp:82-94） | `BLOCK_M,BLOCK_N∈{64,128}`；K=hd 固定（≤256） | ✅ | 速查表 row41 |
| **CANN S2.i=1024 大块**（docs:212） | **Triton 不复刻**：BN 取 64/128/256 + KV 循环，代价=在线刷新每块一次（exp 重缩放）；CANN 1024 是 MTE 启动优化、UB 大、非寄存器友好，属 CANN 特定 | ⚠️ 性能 | 速查表 row41/44；CANN 大块不可直移 |
| Vector 基本块 8×1024、nRatio=8 配比（docs:38-53, tiling:1862） | Triton 无 C/V 显式配比；UB 预算校核等效 tile 字节 | ⚠️ 性能 | 速查表 row34 |
| online softmax + 刷新 `exp[i]=e^{m_{i-1}-m_i}`（docs:11-17） | flash 在线：`m_new=max(m_old,r)`、`acc=acc*exp(m_old-m_new)+dot(exp(r-m_new),v)`（**原生内置**，同构） | ✅ | 速查表 row44/50；闪存算法 |
| 末尾 `÷softmax_sum`（docs:18） | `out = acc / lse`（lse=m_new+log(sum)） | ✅ | 速查表 row44 |
| `softmax_max/sum/out` 三路中间输出（def.cpp:137-156）+ ×8 broadcast（cpu_impl.py:68） | **❌→不需要**：level4 golden 内部重算 softmax、只取 out；省掉 ×8 补位输出 | ❌→不需要 | 结构性更优，少写 3 路 GM ；（速查表 row22 同构） |
| atten_mask select 遮蔽（docs:9） | 本 target 不用；需要时 `tl.where(mask, score, -1e30)`（注意要 ≤ 负极大值压过 softmax） | ⚠️ 正确性 | 速查表 row47；C-031 相关（尾列 ~-inf 经验, crossformer 卡） |
| 因果=sparse_mode=3（P1 §3） | 需要时 `offs_i >= offs_j` 对角 mask（±window 用 `offs` 差值算术带 pre/next_tockens） | ⚠️ 正确性 | size 来自 offs；本 target 不触发 |
| dropout `softmax·drop_mask·(1/keep)`（cpu_impl.py:62） | 需要时 `tl.load(drop_mask)`×`1/keep_prob` 逐元素乘；drop_mask 由调用方预生成（PTA 闭源→ mask bit-exact 不可达，`❌`，同级避免编译期生成） | ⚠️ | 本 target 不触发；随机 mask 生成不是本 kernel 职责 |
| pse（real_shift）与 scale 顺序随 pse_type（def:165, cpu_impl:50/52） | 需要时两态：pse_type 分支选 `(add pse then ·scale)` 或 `(·scale then add pse)`；默认 off | ⚠️ 正确性 | 本 target 不触发 |
| fp32 累加、中间升 fp32（INPUT_T/T, s1s2:767-769） | `q/k/v .to(tl.float32)` → dot → 累加 fp32 → `out_dtype .to()`（RTNE 默认） | ✅ | 速查表 row45；**golden 强制 fp32**（definition reference 用 `.float()`） |
| inner_precise 档 / ImplMode（template_tiling_key.h:17-16） | 主纲强制精确（golden fp32）；高性能够档不选（f16 中间会撞 golden） | ✅ | 与 golden 同舍入（benchmark 纪律） |
| **NZ 格式（Bmm1Format）、NZ→ND置换**（s1s2:869-877,936-940） | `tl.trans(k)` + `tl.reshape` 近似；NZ 拼接逻辑省略 | ⚠️ | 速查表 row49 |
| bmm1 输出经 workspace/GM（tiling_general, docs:45） | **❌→不需要**：Triton tl.dot 板载融合、无需 GM 中转 bmm1 | ❌→不需要 | 结构性更优，少一次 GM 往返 |
| 3-slot CV 流水（Process:861-965）、pingpong（docs:65-74）、SetFlag/WaitFlag | **❌**：不可移植，依赖编译器 multibuffer/num_stages 自动注流 | ❌ | 速查表 row34/35/58 |
| Cube 双发 preload（A2）/preload=3（950）（docs:80-88） | 不显式编码；以 num_stages/并行循环提示编译器 | ⚠️ | 速查表 row57；`num_stages` Triton-Ascend 可能不生效（P2 纪律，须实测） |
| rope/fp8（d_scale_q/k/v,p_scale,sink）/空 tensor | 本 target 不触发；store 为「已知存在、不映射」 | ❌→不需要 | level4 无此输入 |
| enableL1Reuse、BigDoubleBuffer（tilingkey） | 编译器优化域，不映射 | ⚠️ | 速查表 row33/34 |

## 2. BLOCK / 网格设计

**任务轴（S1 行主）**：`query[S(B), H, q_len=S]`，输出 `(B,H,S,hd)`。任务块数 `TB = B*H*cdiv(S,BLOCK_M)`。
**grid**：`grid = (min(TB, aic),)`（`aic`=物理 AIC/Vector 核数，persistent-core，速查表 row13）。kernel 内连续段 `tpb=cdiv(TB,grid)`、`for blk in range(pid*tpb, min((pid+1)*tpb, TB))`。

**BLOCK**（CANN 基本块 → Triton 候选，速查表 row41/44 + P2 纪律「初选偏大须降 BLOCK」）：
| 参 | CANN 依据 | 候选 | 注 |
|---|---|---|---|
| `BLOCK_M`(=S1.i) | 64/128（docs:211, tiling_general:56） | 64, 128 | 高性能够档 128；**D-tier 硬边界（C-032）**：`BLOCK_D>128`（即 head_dim>128）→ 降 `BM=32/BN=64` 腾 192KB UB（p25 实证，对齐 crossformer_v2） |
| `BLOCK_N`(=S2.i) | **1024（CANN 大块）** → Triton 降 64/128/256 | 64, 128, 256 | KV 内层循环；1024 寄存器/UB 反而低效，勿照抄 |
| `head_dim K` | D=hd（≤256, def） | =D 固定 | tl.dot 的 K 维；KV loop 内 k_blk[BN,hd] |
| `BLOCK_S`（M/N 同 S 时无独立量） | — | M=N 由 BM/BN 定 | 自注意力 S1==S2==S |

## 3. Triton 骨架伪码（生成用，稠密 flash 主纲）

```
# wrapper
B,S,D = x.shape; H = n_heads; hd = D//H
SCALE = tl.constexpr(1.0/sqrt(hd))
# ① 4 段投影 = 4×MatMul（与注意力核串接）——level4 接口需先算 q/k/v/out_proj 相关
q = (x @ Wq).view(B,S,H,hd).transpose(1,2)  # [B,H,S,hd]
k = (x @ Wk).view(B,S,H,hd).transpose(1,2)  # [B,H,S,hd]
v = (x @ Wv)...
q,k,v = q.float(), k.float(), v.float()      # fp32 计算域（golden 强制）
q = q.reshape(B*H*S, hd)  # 或按 stride 线性化；kernel 以 (bhi, s) 定位

# kernel(qg, kg, vg, og, SCALE, S:, BLOCK_M=128, BLOCK_N=128, num_warps=...)
pid = tl.program_id(0)
heads = B*H; nblk = cdiv(S, BLOCK_M); TB = heads*nblk
# persistent-core 连续段
tpb = cdiv(TB, tl.num_programs(0))
for tid in range(pid*tpb, min((pid+1)*tpb, TB)):
    bh = tid // nblk; mob = tid % nblk
    o_blk = mob*BLOCK_M
    offs_m = o_blk + tl.arange(0, BLOCK_M)    # query 行
    m_i = tl.zeros([BLOCK_M], fp32) + (-inf); l_i = tl.zeros([BLOCK_M], fp32); acc = tl.zeros([BLOCK_M,hd], fp32)
    q_ = tl.load(qg + offs_m[:,None]*hd + tl.arange(0,hd)[None,:])  # [BM,hd]
    for kv in range(0, S, BLOCK_N):
        offs_n = kv + tl.arange(0, BLOCK_N)
        k_ = load kg [offs_n, hd]   # [BN,hd]
        s_ = tl.dot(q_, tl.trans(k_)) * SCALE          # [BM,BN]
        r  = tl.max(s_, 1); m_new = tl.maximum(m_i, r)
        p  = tl.exp(s_ - m_new[:,None])
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:,None]
        l_i = l_i*alpha + tl.sum(p, 1)
        v_ = load vg [offs_n, hd]   # [BN,hd]
        acc += tl.dot(p, v_)                            # fp32
        m_i = m_new
    out = acc / l_i[:,None]
    store og[offs_m, hd] = out.to(...)                 # 回写 dtype（RTNE 默认）
```

**注**：
1. `tl.dot` 的 K 维 = hd；`p[BM,BN]` 与 `v[BN,hd]` 相点——PV 左手分量随输出 layout（C-030 经验：p 按 (BM,BN)、v 按 (BN,hd) 加载）。
2. 尾块：`tl.load(..., mask=offs<valid, other=0.0)`；V 尾列不参与无需 `~-inf`（稠密无 KV 尾踩空的风险是本映射无 sparse，C-031 仅在带 mask 时关注）。
3. 逐块 KV loop 的 `acc*alpha` 即 CANN 的刷新流程（docs:14）——**Triton 原生在线，不需显式切 SkvSplit 或 workspace**。
4. 骨架必须静态自检（逐 `tid`/`kv`/`mask` 走一遍符号推导）再进 P2.5（README P2 纪律）。

## 4. 精度策略 → 验收

- 主纲强制 fp32 计算域（q/k/v、scores、P、acc），`out` 回写输入 dtype（`.to(dtype)` RTNE）——与 golden（definition reference 全程 `.float()`）同舍入序列。
- **验收**（对齐 level4 evaluator 默认，correctness.py 习惯阈值）：fp32 rtol=2⁻¹⁰/atol=2⁻¹⁶、bf16 2⁻⁶。投影（MatMul）与注意力合起来比 `out_proj(softmax((q·kᵀ)/√hd)·v)`。
- golden 建造纪律（apply_rotary 经验）：Model 方法名避开 nn.Module 保留名（`_apply` 等）；golden 自验走消费方真实 `.to(device).float()` 调用路径。

## 5. 回归：P1 事实去向核对（质量门——无静默丢失）

- scope 覆盖：§0 接口差（投影归位）、§1 表 25 行、§2 BLOCK、§3 骨架、§4 验收。
- 未写进卡的选项已显式标注：`❌→不需要`（softmax 三中间输出 / workspace / rope/fp8 / 空tensor）、`❌`（3-slot 流水 / pingpong / SetFlag/WaitFlag / 显式双缓冲）——均给理由，无静默丢弃。
- 三态均非「未经确认」：P1 每条 tiling/流水/计算事实在 §1 有去向。

## 6. 待 P2.5/P4 实测确认项（两态，不下绝对结论）

- `num_stages`/autotune 在 Triton-Ascend 是否生效（须查 model/zh 支持矩阵或 runbooks，未确认前 skeleton 不带 Trition autotune，防 µs-kernel do_bench 噪声误选）。
- `BLOCK_N` 候选（64/128/256）真实相对快慢；CANN 1024 大块 vs Triton BN 小块的性能差量级。
- online softmax 大 case（S 大、BM×BN 遍历次数多）的地板是否成立（CPU golden vs kernel）。
- 小 shape（自注意力 S 小，如 S≤64）是否 fallback 单块直通（SelfAttention_smallS 分支，见 definition.json）——S 直通单块路径保留，避免整 kernel 循环开销。