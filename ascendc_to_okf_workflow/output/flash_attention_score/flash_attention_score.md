---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: SelfAttention 前向（flash_attention_score）Ascend C → Triton 语义转化卡
description: 批量多头自注意力前向：level4 run(x,n_heads) 内部先做 4 段投影（Wq/Wk/Wv/Wo 的 nn.Linear，seed=42）得 q/k/v，再 softmax(QKᵀ·scale)·V（scale=1/√head_dim）得 z，末段 Wo·z 回 B,S,D。注意力核（本卡映射对象=CANN flash_attention_score）为 CV 融合（Cube matmul ×2 + Vector softmax），Triton 映射为统一批多头 flash 前向：head/query 块作任务轴、key 分块在线 softmax（running max/α 补偿）进内循环；单 K 块情形退化为全 S 单 tile full-softmax。关键决策：4 投影 host fp32 GEMM（kernel 纯注意力）；fp32 计算域 + 末段单次 RTNE；head 视图必须 host 侧 .contiguous()（否则平坦寻址静默错值）；key 尾列 -inf（C-031）；V 按 (N,D) 加载（C-030）；fp32 dot 用 ieee（C-029）。
tags: [attention, cv_fusion, matmul, flash_attention, softmax, self_attention, multi-head, fourelem_proj, sdpa]
confidence: inferred
status: draft
resource: zmm_source_retrieval/src/flash_attention_score（cann/ops-transformer @ d28d8f99a，arch32+arch35）+ 工作流 output/flash_attention_score（manifest/extraction/mapping/p25_report）
created_at: "2026-09-21"
updated_at: "2026-09-21"
---

sources:
  - url: zmm_source_retrieval/src/flash_attention_score/level4_target/SelfAttention/definition.json
    role: primary

# SelfAttention 前向（flash_attention_score）

**总述**：单输入多头自注意力前向。`run(x[B,S,D], n_heads)` 数学五步：
① 4 段投影 q=Wq·x、k=Wk·x、v=Wv·x（seed=42 权重，`nn.Linear(D→D,bias=False)`，Kaiming uniform）；
② scores = QKᵀ·scale，scale=1/√head_dim、head_dim=D/n_heads，按头 reshape (B,H,S,hd)·transpose(1,2)；
③ weights=softmax(scores,dim=-1)；④ z=weights·V（fp32 累加）→(B,S,D)；⑤ out=Wo·z。
**kernel 类型：CV 融合**（Cube matmul ×2 = QKᵀ 与 P·V + Vector softmax）。映射对象的 CANN 参考是
`flash_attention_score` 注意力核本体（`op_kernel/flash_attention_score.cpp:379` KERNEL_TYPE_MIX_AIC_1_2），
其只含 ②~④ 的注意力核；**①投影四段在 kernel 外（=CANN MatMul），本卡由 host fp32 GEMM 承担**（mapping §0 论证 4）。
Triton 注意力核 = 统一**批多头 flash 前向**（在线 softmax 复刻），小 S 情形收缩为单 tile full-softmax
（P2.5 已实测，见 §8/§11）。

## 1. 功能与公式

- 功能：批量多头自注意力前向，输出被 Wo 回投影。
- 计算公式（level4 `definition.json reference` 逐字；flash 在线复刻以等价拆分实现）：
  $$
  \begin{aligned}
  &q[m] = W_q x[m],\ k[m]=W_k x[m],\ v[m]=W_v x[m],\qquad m\in(B,S)\\\\
  &\text{score}[h,m,j] = \tfrac{1}{\sqrt{d}}\,\sum_{d'} q[m,h,d']\,k[j,h,d'],\quad d=\mathrm{head\_dim}\\\\
  &P[h,m,j] = \mathrm{softmax}_j(\text{score}[h,m,\cdot]) = e^{(\text{score}-\max_j)}/\textstyle\sum_j e^{(\text{score}-\max_j)}\\\\
  &z[m,h,d'] = \sum_j P[h,m,j]\,v[j,h,d'],\qquad \mathrm{out}[m] = W_o z[m]
  \end{aligned}
  $$
  无 causal/pse/mask/dropout/varlen（level4 恒规整 BSH 全 softmax）。
- **语义分层**：
  - kernel（Triton）：纯批注意力（②~④ 核心），q/k/v 以行主序连续 [B·H,S,hd] 入。
  - host/投影层：4× nn.Linear = torch fp32 matmul（Wᵀ·x）+ head reshape `.contiguous()`——**不在 kernel**。
  - benchmark：level4 `definition.json` torch golden（含自建投影权重，seed=42）为**单标杆**；投影与 golden 共享同份权重。
- 计算流程：投影(host fp32) → 分头 contiguous → flash：QKᵀ(dot)→×scale→在线 softmax(exp/max/α 补偿)→P·V(dot)→除 l → Wo 投影(host) → 回 B,S,D。
- 锚点：`definition.json reference`（golden）；CANN 内核 `op_kernel/flash_attention_score.cpp:379`（KERNEL_TYPE_MIX_AIC_1_2）、
  `op_kernel/arch32/flash_attention_score_s1s2_bn2gs1.h:85`（主类 ProcessVec1 在线 softmax）；extraction.md §11 对照表。

## 2. 原型与约束

### 输入输出（level4 `definition.json`；CANN def.cpp 输入 q/k/v + 属性 scale_value/head_num 对应，见 extraction §2.1）

| 参数名 | 入参类型 | 必选性 | dtype | format | shape | 语义 |
|---|---|---|---|---|---|---|
| x | 输入 | 必选 | fp16/bf16/fp32 | ND(BSH 连续) | (B,S,D) | token 序列，D=hd·n_heads |
| n_heads | 标量 | 必选 | int | — | — | 头数，按 D 整除分块 |
| （kernel 内 q/k/v） | 输入 | — | 同 x | ND 连续 | (B·H,S,hd) | host 投影后 分头 .contiguous() 产出 |
| （kernel 内 scale） | 标量 | — | float | — | — | 1/√head_dim（CANN scale_value 对应） |
| output | 输出 | — | 同 x | ND | (B,S,D) | Wo·z 回投影后的注意力输出 |

### 属性（CANN def.cpp 与 level4 的差，超范围登记 §8）

| 属性 | 类型 | 默认 | 语义 |
|---|---|---|---|
| scale_value | float | 1/√head_dim | 缩放（level4 恒 1/√head_dim，definition.json `/ math.sqrt(head_dim)`） |
| head_num | int | — | level4 由 n_heads 给出 |
| keep_prob/pre_tockens/next_tockens/sparse_mode/pse_type/input_layout/inner_precise/drop_mask/... | — | — | 超 level4（不翻译，§8） |

### 原型约束

- D % n_heads==0 且 head_dim>0（.view 隐含；CANN def.cpp head_num :161）。
- **head 视图契约（P2.5 后端发现，host 必守）**：`.view(B,S,H,hd).transpose(1,2).reshape(BH,S,hd)` 在 **B=1**
  时返回**非连续交错视图**（stride=(hd, S·hd,1) 实为 [S,H,hd] 布局），kernel 平坦 `base+rows*hd+hdc` 行主序寻址
  会静默读错（B≥2 恰好 gives 行主序才偶然正确）。**必须 `.contiguous()`**（skeleton 已修，p25_report ②门）。
- S、head_dim 均非 2 幂 → BLOCK 上浮 2 幂 + mask 处理（C-002 整型 mask）。
- 拦截校验：无（语义由 workload 给；wrapper 仅 `assert contiguous`）。

## 3. 规格约束

| 类别 | 取值 | 来源锚点 |
|---|---|---|
| 支持 NPU 型号 | Ascend910B/A2、950 等（CANN 9.1.0 arch32/arch35） | manifest / hardware_platforms.md |
| dtype 全集 | fp16/bf16/fp32（输入）；中间 scores/P/o **全 fp32** + 末段一次 RTNE | dtype_support §4 / C-027 精度规则 |
| format | ND（BSH 连续） | level4 workload |
| 确定性计算 | 是（每输出一次写回，无跨核规约/原子，规避 C-024） | §0 结构性论证 |
| 交付形式 | python 直调（Triton 注意力核 + host 投影） | 工作流 |

### 架构差异

- CANN FAS 分 arch32（910B，CV 流水六阶段）/arch35（950 regbase）；Triton 无显式 UB/L0C/flag → 降级编译器
  隐式流水（mapping §5）。回放只需语义，不照搬调度时隙。

## 4. 分核策略（→ Triton grid）

- Ascend C 事实：FAS tiling 任务轴 B×N(head)×S1(query 块)，S2(key) 内循环`GetS1LoopRange/GetS2LoopRange`；
  usedCoreNum 分核（extraction §4）。
- **Triton 决策（统一 flash 前向）**：
  - **任务轴 = (B·H · cdiv(S, BLOCK_M))** —— head/query 块做任务（输出驻留轴），key 分块进 flash 内循环。
    大 S（S 至 897）必须 query 也分块以控 UB；小 S（S≤~59）cdiv=1 → 每 head 一任务、BLOCK_N≥S 单 key 块
    → 收缩为 P2.5 已实测的单 tile full-softmax。
  - 任务块总量公式：`NUM_BLOCKS = B · H · cdiv(S, BLOCK_M)`。
  - 核分配范式：persistent-core（grid-stride 交织）`for task in range(pid, NUM_BLOCKS, num_progs)`；
    `grid = min(NUM_BLOCKS, num_core)`，num_core 运行时 `get_device_properties` 查询，**禁写死**（C-011/
    hardware_platforms §3）。
  - **P2.5 实测特例**：小 S（BLOCK_M≥S、SPADS=1）已证 `grid=(1,)` 串行 head 循环 + 单 K 块 exact
    （B=1 亦稳定，见 p25_report）。对全 workload 的推广即上文统一 tiled 形态（与 crossformer_attention 同构）。

## 5. BLOCK / Tiling 参数

### 映射

| tiling 字段 | 语义 | Triton 对应 | 候选值 |
|---|---|---|---|
| S1TemplateNum∈{0,64,128,512} | query 轴 | BLOCK_M constexpr | {64,128}（S≤64 时取 BLOCK_M≥S → cdiv=1） |
| S2TemplateNum∈{0,128,256,512} | key 轴 | BLOCK_N constexpr | {64,128}（S≤BLOCK_N → 单 K 块 full-softmax；大 S 分块在线） |
| DTemplateNum∈{0,64,128,256} | 特征维 | BLOCK_D = next_pow2(head_dim) | head_dim∈[32,256]→{32,64,128,256}；>128 动态 D（CANN DTemplateType=0，144/224/256 不命中 D 档）；恒满足 C-010 |
| scale_value | 缩放 | scale fp32 标量（runtime，C-013） | 1/√head_dim |

- BLOCK_M=BLOCK_N=BLOCK_D=128 覆盖主 S/head_dim；head_dim∈{224,256} 需 BLOCK_D=256（mask<hd 原样）。
- 大 S（主目标 100% S>64）：BLOCK_M/N 分块 + 在线 softmax，必走 `for s2` K 循环（running max/α 补偿）。
- 小 S（次要分支）时 BLOCK_N≥S 使在线 for 循环仅 1 迭代 → 与单 tile full-softmax 等价（P2.5 初轮验证）。

## 6. 片上内存预算（950 UB 248KB 可用）

| BLOCK / dtype | 主要 tile（fp32 计算域） | 峰值 | 占 248KB |
|---|---|---|---|
| BM128/BN128/BD128，fp16 输入 | q/kt/v 各 32KB；s 64KB；p~64KB；o acc 64KB | ≈288KB | 贴顶 ⚠️（性能风险；multibuffer 未含）→ 大 shape 降 BM/BN=64 |
| BM64/BN64/BD128，fp16 | 各半 | ≈144KB | ≈58% 安全 |
| BM64/BN64/BD64（主形状 hd≤64） | 进一步减半 | ≈72KB | ≈29% 安全 |
| **fp32 输入** | 输入 tile ×4（s/p/o 仍 fp32，输入 q/k/v 翻倍） | ≥1.5× | 大 shape 必降 BM/BN=64 保安全 |

- 全 S 单 tile（小 S 路径）峰值见 skeleton 注释：S64×hd128≈176KB≈71% ⚠️；主形状（hd≤64）≤45% 安全。
- 对齐：BLOCK_M/N/D∈{64,128} 恒 ≥32B（C-010）；store mask 整型（C-002）。
- loop-carried 累加器初值**禁 `tl.zeros`**（C-016）：全 False mask load 产零/产 -inf + 显式 `.to(tl.float32)`。

## 7. 流水编排与 Triton 可达性

### 流水结构

flash 外循环（key 分块）内：load K/Vᵀ → dot(q,kᵀ)×scale → 在线 softmax（running max/α 补偿/l 更新）→
dot(p·v) 累加 o → 除 l → Wo 投影（host）。单 kernel 完成注意力核。

### 可达性判定表

| CANN 手段 | 锚点 | 可达性 | Triton 写法 / 降级 | 预期影响 |
|---|---|---|---|---|
| Cube bmm1↔Vector softmax↔Cube bmm2 CV 换手 | `flash_attention_score.cpp:379` MIX_AIC_1_2 | ❌ 不可达（CV flag） | 编译器隐式流水（load→dot→softmax→dot 软流水 + multibuffer） | 达不到手工 ping-pong；单 tile 收益有限（P2.5 未测量） |
| 在线 softmax 多块重标度 | arch32 主类 ProcessVec1 :155 | ⚠️ 部分（正确性补足） | 手动 running max + α=exp(m_old−m_new) 在线维护（crossformer 同法） | 数学等价；向量化由编译器 |
| key 尾列遮蔽（越界 −inf） | C-031（flash 注意force） | ✅ 必须 | `s = tl.where(n_ok[None,:], s, -inf)`（单 tile 同样，skeleton :68） | 防 softmax 假质量 |
| p@V 的 V 加载方向 | C-030 | ✅ 必须 | V 按 (BLOCK_N,BLOCK_D)；QKᵀ 的 K 用 tl.trans 或 (D,N) 镜像 | 防 V 转置错值 |
| TQue/TBuf double buffer | 各 Que/Buf[2] | ⚠️ 编译器自动 | multibuffer 默认开；**禁 num_stages/num_warps**（C-006） | 性能两态 |
| 4 投影 / Wo 回投影 | def.cpp 无（kernel 外） | ✅ host | torch fp32 matmul（host GEMM），共享 seed=42 权重 | 语义隔离注意力核 |
| nd2nz / NZ 格式 | — | ❌→不需要 | Triton 无 NZ，寄存器直通 dot | — |

## 8. Triton 骨架（生成用伪码）

```python
# 可运行小 S 实现见 output/flash_attention_score/triton_skeleton.py（P2.5 8/8 全 PASS，单 K 块路径）。
# 以下为统一 tiled flash 前向（大 S 推广，与 crossformer_attention 同构）；BLOCK_N≥S 时内层仅 1 迭代=单 tile。
IS_FP32 = (input dtype == fp32)     # wrapper 据 dtype 传 constexpr（禁 kernel 内 dtype 分支，C-023②）
OUT_TY  = <output element_ty>

@triton.jit
def _sa_attn_fwd_kernel(q,k,v,o, B,S,HD,scale, IS_FP32, OUT_TY,
                        BLOCK_M,BLOCK_N,BLOCK_D):
    pid = tl.program_id(0)
    num_m = cdiv(S,BLOCK_M); NTOT = B*H*num_m        # H 以运行时标量传入（免 kernel 内 H）
    dcol = tl.arange(0,BLOCK_D); d_ok = dcol < HD
    for task in range(pid, NTOT, tl.num_programs(0)):  # persistent grid-stride（C-011）
        bh, mblk = task//num_m, task − bh*num_m         # 免 %（C-001）
        base = bh * S * HD
        mrow = mblk*BLOCK_M + tl.arange(0,BLOCK_M); mm = mrow < S; row = base + mrow
        q = tl.load(q_ptr + row[:,None]*HD + dcol[None,:], mask=mm[:,None]&d_ok[None,:], other=0.0).to(tl.float32)
        o_a = tl.load(q_ptr+row, mask=(mrow[:,None]>S), other=0.0).to(tl.float32)          # 产零（C-016）
        l_i = tl.load(q_ptr+row, mask=(mrow>S), other=0.0).to(tl.float32)
        m_i = tl.load(q_ptr+row, mask=(mrow>S), other=float("-inf")).to(tl.float32)
        for n0 in range(0, S, BLOCK_N):                # key 分块（在线 softmax）
            ncol=n0+tl.arange(0,BLOCK_N); n_ok=ncol<S; nrow=base+ncol
            kt = tl.load(k_ptr+nrow[None,:]*HD+dcol[:,None], mask=n_ok[None,:]&d_ok[:,None], other=0.0)  # (D,N)
            v  = tl.load(v_ptr+nrow[:,None]*HD+dcol[None,:], mask=n_ok[:,None]&d_ok[None,:], other=0.0)   # (N,D) C-030
            s  = tl.dot(q, kt, input_precision="ieee" if IS_FP32) * scale
            s  = tl.where(n_ok[None,:], s, float("-inf"))     # C-031 尾列 −inf
            m_c = tl.max(s,1); m_n = tl.maximum(m_i,m_c); alpha = tl.exp((m_i−m_n).to(tl.float32))
            p   = tl.exp(s − m_n[:,None]); l_i = l_i*alpha + tl.sum(p,1)
            o_a = o_a*alpha[:,None] + tl.dot(p.to(OUT_TY), v, input_precision="ieee" if IS_FP32)
            m_i = m_n
        out_t = (o_a / tl.maximum(l_i,1e-24)[:,None]).to(OUT_TY)
        tl.store(o_ptr+row[:,None]*HD+dcol[None,:], out_t, mask=mm[:,None]&d_ok[None,:])
```
wrapper 要点：grid=min(B·H·num_m, num_core)（num_core 运行时查询，禁写死）；**head 视图 [B,H,S,hd]→[BH,S,hd]
必须 `.contiguous()`**（地址契约，§2）；IS_FP32 constexpr；scale runtime fp32。投影四段 = host torch matmul
（Wᵀ·x，seed=42 kaiming 权重，与 golden 同源共享）——**勿并入 kernel**（回放隔离注意力核正确性）。

## 9. 精度标杆与验收

- 标杆类别：单标杆 = level4 `definition.json run()` 逐字移植（host 投影 + fp32 softmax/attn，`golden_run`）。
- 标杆实现要点：seed=42 权重；q/k/v 投影 fp32；scores 全 fp32；softmax(dim=-1)；Wo 回投影。
- **算子级验收标准**（分 dtype，P2.5 ②门恰用）：**fp32**：max_abs≤1e-3 且 bad_ratio≤1e-3（ieee 从严）；
  **fp16/bf16**：`|a−b|>max(0.005·max(|a|,|b|), 2.5e-5)` 判挂，bad_ratio>0.005 或 max_abs>0.1 判 FAIL。
- 计算域：q/k/v 投影与 scores/P/o 全 fp32；末段 store `.to(out_dtype)` 一次 RTNE（对齐 fp32 golden 逐位）。
- **benchmark 语义超集**：workload 50 case 全覆盖（见 §11 包络表）——**主目标 8_SelfAttention（统一大 S）**：
  head_dim∈[32,256]（BLOCK_D 上浮，>128 动态 D）、S∈[512,5182]（100% 走分块在线 softmax）、B∈[1,8]、
  d_model≤8192、三 dtype。小 S 单块分支 archive 于 `SelfAttention_smallS`（§11 注）。
- 典型用例：`(B∈[1,8], S∈[512,5182], D∈[160,8192], n_heads∈[3,64], head_dim∈[32,256])`，三 dtype，100% 大 S。
- 边界：head_dim 非 2 幂（BLOCK_D mask）、S 非 2 幂（BLOCK_S mask）、key 尾列 −inf（C-031）、**B=1 的
  head 视图非连续陷阱（§2，最易静默错值）**。

## 10. Agent actions（生成指引）

1. 识别：单输入多头自注意力前向 = 4 段 host 投影 + batch flash 注意力核（Cube QKᵀ 与 P·V + Vector softmax）→ **CV 融合**。
2. **投影四段在 host（torch fp32 matmul，seed=42 权重）**，kernel 只吃 q/k/v——勿把 GeLU/投影并进 kernel
   （回放连 golden 共享权重，隔离注意力核正确性）。Wo 回投影亦 host。
3. 切分轴取 (B·H·cdiv(S,BLOCK_M))，key 分块进 flash 内循环在线 softmax；persistent grid-stride，grid=min(NTOT,num_core)
   （num_core 运行时查，禁硬编码，C-011）。
4. BLOCK 候选取 BLOCK_M/N/D∈{64,128}；head_dim 非 2 幂 → BLOCK_D 上浮 + `d<HD` mask；UB 大 shape 降 64（§6）。
5. 精度提升点：scores/P/o 累加保留 **fp32**；fp32 输入 dot `input_precision="ieee"`（C-029：f16/bf16 不写 precision，禁写 tf32）。
6. **必做（静默错值，最难抓）**：
   - head 视图 `.contiguous()`（§2，B=1 非连续陷阱）；
   - key 尾列 `−inf`（C-031）；
   - p@V 的 V 按 `(N,D)` 加载（C-030，勿沿用 Kᵀ 的 (D,N)）。
   - 在线 softmax 手动维护 running max + α=exp(m_old−m_new)（金标准小 S 即单 K 块、无 α 也无妨，但大 S 必须）。
7. loop-carried 累加器初值**禁 `tl.zeros`**（C-016）：全 False mask load 产零/产 -inf + 显式 `.to(fp32)`。
8. 勿复刻：FAS 的 S1S2 时隙/QK 双 buffer 等硬件手段（Triton 单 kernel 已完成）；ao 不在 kernel（host）。
9. 验收按 §9 分档容差；唯一标杆 = definition.json run()（fp32 golden），非 CANN 实现。

## 11. 复现性声明

本卡由 level4 golden + CANN `flash_attention_score` 结构转化（confidence: inferred）。P2.5 真机
Ascend910 + CANN 9.1.0 + torch_npu：①编译门 24/24 配置全过（BLOCK_S/BLOCK_HD 代表面 + hd=65→128 档）；
②正确性门**8/8 bad=0.0**（初轮小 S 单 K 块 smoke + corner）——该次实测包络为小 S 单 K 块（S≤59、BLOCK_N≥S）。
**主目标已换为 8_SelfAttention（统一大 S，S∈[512,5182]、head_dim 至 256、100% S>64）**:大 S 必须的
S1S2 分块 tiled 在线 softmax 为本次重构 `triton_skeleton.py` 的主路径（running max/α 补偿 + K 循环，与
crossformer_attention 已验证形态同构，C-029/030/031 同源）；head_dim∈{144,224,256} 超 D 模板上限 → 动态 D。
**大 S 骨架已上板重跑 P2.5（Ascend910_9372）：①编译门 + ②正确性门全过**——5 组 big-S smoke（3 dtype × HD48~128）
bad_ratio=0.0（fp32≈4e-8、bf16/fp16 ≤2.4e-4，远小于阈值），CPU 在线-softmax 自检 6/6 亦过。验证中修复两处
重写引入的后端 bug（写卡关键，p25_report 〇）：**①head 基址勿多乘 HD**（`base=bh·S` 而非 `bh·S·HD`——BH≥2 时
多乘 HD 直接越界，aicore MTE；BH=1 时宜被掩盖）、**②fp32 计算域下 kt/v 必须 `.to(fp32)`**（否则 `tl.dot`
「Got fp32 and fp16」编译断言，仅 fp16/bf16 输入触发）。小 S 单块分支保留为次要（archive `SelfAttention_smallS`）。
回放验证：**未做（P4 双盲回放未执行）**——confidence 保持 inferred。

## 12. 关联

- **同家族前向**：`output/crossformer_attention/crossformer_attention.md`（本卡注意力核 = 其无窗口分组的
  批多头 flash 特例，共享在线 softmax / C-029/030/031 机制）；反向卡 `flash_attention_score_grad`。
- **契约**：C-001（禁 %）、C-002（mask 整型）、C-006（num_stages 无效）、C-011（persistent grid + 核数运行时查）、
  C-016（禁 tl.zeros 累加器初值）、C-023②（constexpr dtype）、C-029（fp32 ieee / 禁 tf32）、C-030（V 定向）、
  C-031（key 尾列 −inf）。**host 地址契约**（P2.5 新增，非后端缺陷）：head 视图须 `.contiguous()`，见 §2。
- **材料**：mapping.md（P2）→ p25_report.md（P2.5）→ validation_report.md（P4 补）。包络更正：mapping §0
  原写 S≤59 过窄，实测 workload S 至 897 → 本文 §5/§11 以统一 tiled 形态覆盖。