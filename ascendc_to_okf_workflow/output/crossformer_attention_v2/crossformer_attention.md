---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: CrossformerAttention（分组/窗口化多头注意力）Ascend C → Triton 语义转化卡（v2）
description: CrossFormer 窗口化多头自注意力——token 序列（可选 reshape 为 H×W）按 (gh,gw) 窗口分组，组内 QKᵀ·scale+加性 pos_bias/mask→softmax→·V。CV 融合，映射为标准批多头 flash（在线 softmax 复刻）. v2 核心：BLOCK 几何绑定 CANN arch35 host 档位阶梯（D-tier AlignUp(D,64)→{64..768}、S1/S2 默认 128/128），并修原骨架硬编码 BLOCK_D=128 ⇒ 单头大 C 算错缺陷。
tags: [attention, cv_fusion, matmul, flash_attention, softmax, grouped, windowed, crossformer, multi-head, host_tiling_ladder]
confidence: inferred
status: draft
resource: zmm_source_retrieval/src/flash_attention_score（cann/ops-transformer @ d28d8f99a）+ src/flash_attention_score_grad 工作流 output/crossformer_attention_v2
created_at: "2026-10-10"
updated_at: "2026-10-10"
---

sources:
  - url: zmm_source_retrieval/src/flash_attention_score/level4_target/CrossformerAttention/definition.json
    role: primary
  # 说明：双副本（workflow output/crossformer_attention_v2/ 为源，KB 版如需入库须 source_family=ops、
  # sources[].url=GitCode @sha 永久链接、role=example）。本卡为 workflows 源卡，未入 KB。

# CrossformerAttention（分组/窗口化多头注意力）v2

**总述**：对 token 序列做（可选窗口化的）多头 scaled dot-product 注意力。语义 = level4 `definition.json` torch
golden：`qh·khᵀ·scale(+pos_bias)(+mask) → softmax(-1) → ·vh`；窗口分组是 **host 侧 layout 变换**，kernel 退化为
**标准批多头 flash 注意力**。**kernel 类型：CV 融合**（Cube matmul ×2 + Vector softmax）。CANN 参考 =
`flash_attention_score` 的 pse_shift/atten_mask/scale_value 路径 + FlashSoftmaxV2 在线重构。
**v2 较 v1 增量**（工作流 v1.5 CANN 价值方法论）：① 补取 **arch35 host 档位阶梯**（D-tier/S1-S2 默档）——
v1 只取 tiling 任务轴与 S1/S2 语义，漏了「默认档位」这一 L1 最优值；② 修 **BLOCK_D 硬编码 128** 缺陷
（单头 C=1024 时 v1 只算前 128 特征 = 静默错值）；③ 实证 **大 D>128 的 192KB UB 溢出边界 + BLOCK_M=32 逃逸**。

## 1. 功能与公式（详设 §1.1）

- 功能：窗口内多头自注意力，含加性位置偏置与掩码。
- 公式（level4 reference 反推，flash 在线 softmax）：
  $$
  score[h,m,j] = scale\cdot\sum_d q[m,h,d]\cdot k[j,h,d] + pos\_bias[h,m,j] + mask[h,m,j]
  $$
  $$
  P[h,m,j] = softmax_j(score[h,m,\cdot]) = e^{(score-\max_j)}/\textstyle\sum_j e^{(score-\max_j)}
  $$
  $$
  out[m,h,d] = \sum_j P[h,m,j]\cdot v[j,h,d]
  $$
  scale = 属性或 `head_dim^{-0.5}`；head_dim = C/n_heads。窗口分组把 N(=H·W) 分成 N_w=(H/gh)·(W/gw) 个
  G=gh·gw 窗口，窗内 token 按本地索引 (lh,lw)→lh·gw+lw 排布。
- **语义分层**：kernel=纯批注意力（输入 (N_w·G,C) 窗口连续）；host/layout=窗口分组 `idx`/`inv`（wrapper torch
  gather，**不是 kernel**）；benchmark=level4 `definition.json` reference 单标杆。
- 锚点：`docs/FA算子设计介绍.md:1-19`（flash 4 步）；`docs/aclnnFlashAttentionScore.md:23`（公式）。

## 2. 原型与约束（详设 §2 / §2.1）

### 输入输出（来源：level4 `definition.json`）

| 参数名 | 入参类型 | 必选性 | dtype | format | shape | 语义 |
|---|---|---|---|---|---|---|
| x | 输入 | 必选 | fp16/bf16/fp32 | ND | (M,C)/(B,M,C) | token 序列/空间展平(self q/k/v) |
| q/k/v | 输入 | 可选 | 同 x | ND | (M,C) | 独立 q/k/v，缺省=x |
| group_size | 标量 | 必选 | int/[int,int] | — | (gh,gw) | 窗口大小 |
| num_heads | 标量 | 可选 | int | — | — | 头数，缺省 1 |
| mask | 输入 | 可选 | fp16/fp32 | ND | (H,N,N)/(N,N) 或 (H,G,G) | 加性掩码（窗口局部） |
| pos_bias | 输入 | 可选 | fp16/fp32 | ND | 同 mask | 加性位置偏置（窗口局部） |
| scale | 标量 | 可选 | float | — | — | 缩放，缺省 head_dim^-0.5 |
| feature_shape | 标量 | 可选 | [H,W] | — | — | x 空间形状用于分组前 reshape |
| output | 输出 | — | 同 x | ND | (M,C) | 注意力输出 |

### 原型约束

- head_dim=C//num_heads 须整除（torch `.view` 隐含）；CANN 对应 head_num（`flash_attention_score_def.cpp:161`）。
- 分组要求 (H%gh)==0、(W%gw)==0（reference 不 pad）。
- ⚠️ **level4 reference 缺陷**：feature 分支全序列 mask/pos_bias 用于窗口 attention 会 shape 不广播报错。
  本卡契unr：mask/pos_bias 按**窗口局部坐标** (H_heads,G,G) 应用（feature 无 bias/单窗口带 bias 两干净路径与
  reference 严格一致；组合 case 以本卡契约为准）。
- 非 feature 时 group_size 被忽略（全序列单窗口）。
- **head_dim 契约（v2）**：C 可达 1024、n_heads 可小 → head_dim∈{24..1024}。Triton BLOCK_D 须 ≥head_dim
  （v1 硬编码 128 单头大 C 会静默丢特征）。大 D 的 UB 约束见 §6。

## 3. 规格约束（详设 §3）

| 类别 | 取值 | 来源锚点 |
|---|---|---|
| 支持 NPU 型号 | Ascend910B/A2、9372；950 等（CANN 9.1.0） | hardware_platforms.md |
| dtype 全集 | fp16/bf16/fp32（中间 attn/o fp32 累加） | level4 workload / def.cpp |
| format | ND | level4 workload |
| 确定性计算 | 是（单 window/head 独立） | 骨架设计 |
| 交付形式 | python 直调（Triton） | 工作流 |

### 架构差异
FAS 参考 arch32（910B KERNEL_TYPE_MIX_AIC_1_2）/ arch35（DAV_3510 regbase，910b 生产路径）两套。
level4 与 Triton 骨架取算法结构（flash 重构），不照搬 slot 时序。**regbase 深流水（taskId ring/L1 3-buffer）
不可移植**（v1.5 L2 排除清单）。

## 4. 分核策略（详设 §4.1.1 → Triton grid）

- Ascend C：FAS `tiling.cpp` 任务轴 B×H×S2，usedCoreNum 分核，Skv 循环进核内。
- **Triton 决策**：
  - 任务轴 = (N_w · n_heads · cdiv(G, BLOCK_M))（query 输出轴做任务，key 分块进 flash 外循环）。
  - `NUM_BLOCKS = N_w·n_heads·cdiv(G,BLOCK_M)`；persistent-core grid-stride：
    `grid = min(NUM_BLOCKS, num_core)`，`for task in range(pid, NUM_BLOCKS, NUM_CORE)`（CV 取 Cube 核数 C-011）。

## 5. BLOCK / Tiling 参数（详设 §4.1.2 / §4.1.3）—— v2：绑定 CANN host 档位阶梯

### v2 主证据：arch35 host 档位阶梯（v1.5 第一顺位）

CANN `op_host/arch35/flash_attention_score_tiling_basic.cpp`：

- **D 轴分段 `CalcDBasicBlock`(:51)**：`dBasicBlock = AlignUp(dSize + dSizeRope, 64)` → {64,128,192,256}；
  **>256 → 768**。→ **D-tier = {64,128,192,256,768}**，Triton 化 = `BLOCK_D = ≥head_dim 最小 2 幂`。
- **S1/S2 主档 `CalcS1S2BasicBlock`(:79)**：
  - ① fp8 空态 → S1=128,S2=512；② D=64 DN 优化（无 mask/pse/dropout，非 fp32/fp8）→ 128/256；
  - ③ D>256 → fp32? **64/128** : 128/128；④ 缺省（D≤256）→ **128/128**。
  - **Level4 主流（f16/bf16/fp32, head_dim≤256, 常带 mask/pos_bias）命中④ → CANN 默认档 = 128/128**。
- **关键现实（v1.5 L1-unlock/L2-bound）**：CANN 128/128 靠 **L1 3-buffer + D-workspace**（大 D bmm2 落 GM,
  `GetWorkspaceSize`:180）撑起；Triton 无此机制 → **128/128 复刻会 192KB UB 溢出**。故 Triton 取
  **64/128 可移植近邻**（v1.2 到 v2 一直如此，但 v2 给了 CANN 依据 + UB 硬证，不再靠猜）。

### 映射表

| tiling 字段 | CANN 档位（锚点） | Triton 对应 | 候选值 |
|---|---|---|---|
| D（head_dim） | D-tier AlignUp(D,64)→{64,128,192,256}/768 | BLOCK_D=≥head_dim 最小 2 幂 | {128,256,512,1024} + d<HEAD_DIM mask |
| S1（key 分块） | 默认 128 | BLOCK_N | 128（head_dim≤128）；64（大 D 逃逸）|
| S2（query 任务） | 默认 128 | BLOCK_M | 64（head_dim≤128）；32（大 D 逃逸）|

**BLOCK_D>128 自动降 BLOCK_M=32/BLOCK_N=64**（fp32 O 累加器主导 UB，见 §6）。D>256 需 D-split（登记边界）。

## 6. 片上内存预算（详设 §5）—— v2 修正 + UB 溢出实c证

| 配置 | buffer | 字节 | 校核（192KB）|
|---|---|---|---|
| head_dim≤128（primary）| q16K + k/v 各 16K + s/p 各 32K + o_i 32K | ≈112KB | ✅ ≈60% |
| head_dim=192/256（BLOCK_M=32/N64/D256）| q16K + k32K + v32K + s/p 各 8K + o_i 32K | ≈128KB | ✅（BLOCK_M=32）|
| head_dim=192/256（BLOCK_M=64）| o_i=64K + s32K + p32K + ... | **>192KB** | ❌ **UB overflow**（p25 实证）|

**v2 实证（v1.5「排除法≠穷举」）**：head_dim>128 在 BLOCK_M=64 时
`ub overflow, requires 1587200 bits while 1572864 bits available`（超 ~4336 bits，f16 hd192/hd256；
fp32 超 ~8000）。编译器自动多缓冲/code-motion 关闭仍溢出。**根因：fp32 O 累加器（BLOCK_M×BLOCK_D）
随 BLOCK_D 线性暴涨，BLOCK_N 影响甚微**。→ 逃逸 = 降 BLOCK_M（O 减半）；已验证 BLOCK_M=32/hd256 → rmse=5e-8。

**结论**：大 D（BLOCK_D>128）的模型 = fp32 累加器放不下 → canonical 近邻（64/128）只覆盖 head_dim≤128；
更大 head_dim 走 BLOCK_M=32 或 D-split（对齐 CANN D-workspace 语义）。这是 **L2 边界**，非降级逃避。

## 7. 流水编排与 Triton 可达性（详设 §6）

flash 外循环（key 分块）内：load K/Vᵀ → dot(q,kᵀ)×scale(+bias) → 在线 softmax（max/exp/α 补偿/l 更新）→
dot(p·v) 累加 o。单 kernel 完成 Q 块，无跨块规约。

| level4/CANN 手段 | 锚点 | 可达性 | Triton 写法 / 降级 |
|---|---|---|---|
| FlashSoftmaxV2 在线重构 | FA设计介绍.md §1 | ⚠️ 部分（正确性补足）| tl.dot + loop-carried maximum/exp/sum + α=exp(m_old−m_new) |
| key 尾列遮蔽→-inf | atten_mask 布尔 select | ✅ | `s = tl.where(n_ok, s, -inf)`（**必须**，C-031）|
| 加性 pos_bias/mask | FAS pse/atten_mask | ✅ | bias tensor (n_heads,G,G) 全窗共享，s+=bias |
| double buffer 掩盖 | — | ⚠️ 编译器自动 | num_stages/multibuffer（C-006 运行时无效）|
| 窗口分组重排（host）| — | ✅（不在 kernel）| wrapper idx/inv gather |
| 大 D D-workspace（bmm2 落 GM）| arch35 GetWorkspaceSize | ❌ | 需自建 D-split（登记）|
| regbase 深流水（L1 3-buffer/taskId）| arch35 `_tiling_basic.cpp` | ❌ | 排除清单（v1.5 L2）|
| 原生在线 softmax 指令 | — | ❌ | 显式循环复刻 |

## 8. Triton 骨架（生成用伪码，P2.5 验证过）

```python
# 可运行实现见 output/crossformer_attention_v2/triton_skeleton.py（P2.5 主核 7/7 PASS + hd256 逃逸 PASS）。
# 关键 v2 决策：BLOCK_D 由 wrapper 按 head_dim 动态（bd = next_pow2_head_dim），bd>128 时 BLOCK_M/N 自动降。
@triton.jit
def _crossformer_attn_kernel(q,k,v,bias,out, NW,G,C,HEAD_DIM,N_HEADS,scale,STRIDE_T,
                             HAS_BIAS,IS_FP32,OUTPUT_DTYPE:constexpr, BLOCK_M,BLOCK_N,BLOCK_D):
    pid = tl.program_id(0)
    num_m = (G + BLOCK_M - 1)//BLOCK_M; NTOT = NW*N_HEADS*num_m
    dcol = tl.arange(0,BLOCK_D); d_mask = dcol < HEAD_DIM
    for task in range(pid, NTOT, tl.num_programs(0)):      # persistent grid-stride
        blk=task%num_m; rem=task//num_m; head=rem%N_HEADS; w=rem//N_HEADS
        w0=w*G; feat=head*HEAD_DIM; foff=feat+dcol
        mrow=blk*BLOCK_M+tl.arange(0,BLOCK_M); m_ok=mrow<G; row=w0+mrow
        qoff = q + row[:,None]*STRIDE_T + foff[None,:]
        q = tl.load(qoff, mask=m_ok[:,None]&d_mask[None,:], other=0.0)
        o_i = tl.load(qoff, mask=(mrow[:,None]>G), other=0.0).to(tl.float32)  # C-016
        rowv = w0+tl.arange(0,BLOCK_M)
        l_i = tl.load(q+rowv, mask=(mrow>G), other=0.0).to(tl.float32)
        m_i = tl.load(q+rowv, mask=(mrow>G), other=float("-inf")).to(tl.float32)
        for n0 in range(0, G, BLOCK_N):
            ncol=n0+tl.arange(0,BLOCK_N); n_ok=ncol<G; nrow=w0+ncol
            kt = tl.load(k+nrow[None,:]*STRIDE_T+foff[:,None], mask=n_ok[None,:]&d_mask[:,None], other=0.0)  # (BLOCK_D,BLOCK_N)
            v  = tl.load(v+nrow[:,None]*STRIDE_T+foff[None,:], mask=n_ok[:,None]&d_mask[:,None], other=0.0)  # (BLOCK_N,BLOCK_D) C-030
            s = _dot(q, kt, IS_FP32)*scale
            s = tl.where(n_ok[None,:], s, float("-inf"))     # C-031 尾列 -inf
            if HAS_BIAS:
                b = tl.load(bias+head*G*G+mrow[:,None]*G+ncol[None,:], mask=m_ok[:,None]&n_ok[None,:], other=0.0)
                s += tl.where(n_ok[None,:], b, float("-inf"))
            m_c=tl.max(s,1); m_n=tl.maximum(m_i,m_c); alpha=tl.exp((m_i-m_n).to(tl.float32))
            p=tl.exp(s-m_n[:,None]); l_i=l_i*alpha+tl.sum(p,1)
            o_i=o_i*alpha[:,None]+_dot(p.to(OUTPUT_DTYPE), v, IS_FP32); m_i=m_n
        out_t=(o_i/tl.maximum(l_i,1e-24)[:,None]).to(OUTPUT_DTYPE)
        tl.store(out+row[:,None]*STRIDE_T+foff[None,:], out_t, mask=m_ok[:,None]&d_mask[None,:])
# wrapper：bd = next_pow2(head_dim); BM,BN = (64,128) if bd<=128 else (32,64);  grid=min(...); o[_inv]
```

## 9. 精度标杆与验收（详设 §测试设计）

- 标杆类别：单标杆 = level4 `definition.json reference`（torch fp32 重算 softmax）。
- 验收标准：对 golden `mae/rmse ≤ 2e-2`（f16/bf16）或 `≤1e-3`（fp32）；形状一致。
  P2.5 实测 f16 1.3e-4 / bf16 4.4e-6 / fp32 3.1e-8 / hd256逃逸 2.7e-8。
- benchmark 语义超集：mask/pos_bias 窗口局部切片（reference 缺陷扩展）；workload 50 case。
- 典型用例：`(M=49..7921, C=96..1024, n_heads=1..512, G≤7921)`。
- 边界：G 尾块 -inf、head_dim 非 2 幂（BLOCK_D mask）、fp32 ieee（C-029）、大 D（§6 边界登记）。

## 10. Agent actions（生成指引）

1. 识别：分组/窗口化多头注意力，CV 融合（Cube 两 matmul + Vector softmax）→ 映射**标准批多头 flash 注意力**。
2. 切分轴 (N_w·n_heads·num_m)，key 分块进 flash 外循环；persistent grid-stride（勿「一块一 program」）。
3. **BLOCK 依 CANN host 档位阶梯（v2 核心）**：`BLOCK_D = next_pow2(head_dim)`（D-tier 缩放，禁硬编码 128——
   v1 单头大 C 会静默错值）；head_dim≤128 → BLOCK_M=64/BLOCK_N=128；head_dim 128~256 → **BLOCK_M=32/N64**
   （192KB UB 溢出实证，BLOCK_M 是 fp32 O 累加器主导项）；head_dim>256 → **D-split 登记未实现**。
4. 精度：attn s 与 o 累加保 **fp32**；fp32 输入 dot `input_precision="ieee"`（C-029，f16/bf16 不写，禁 tf32）。
5. **不可达/必做**：窗口分组在 wrapper 做（host gather idx/inv），kernel 连续仿射 load；flash 在线 softmax 手动
   维护 running max/α 补偿（无原生）；**key 尾列必须 -inf（C-031）**。
6. **勿复刻**：FAS S1S2 时隙/QK 双 buffer/regbase 深流水/L1 3-buffer/D-workspace——Triton 单 kernel 已完成或不可移植（v1.5 L2）。CANN 128/128 默档是 L1 结构解锁，但需自证 UB（192KB 复刻溢出）。
7. `p@V` 的 V 定向 `(BLOCK_N,BLOCK_D)`（key×feature），勿误用 Kᵀ 的 (D,N)（C-030）。
8. 验收按 §9（f16/bf16 2e-2，fp32 1e-3）。

## 11. 复现性声明

v2 卡由 level4 golden + CANN FAS 结构 + **arch35 host 档位阶梯**（v1.5 方法论）转化（confidence: inferred）。
P2.5 ①编译门+②正确性门已在真机 Ascend910（arch 9372）+ CANN 9.1.0 + torch_npu 2.10 跑通：
主核 7/7 PASS（head_dim≤128，rmse 见 §9）+ v2 专项 hd256 逃逸 PASS（BLOCK_M=32/N64/D256, rmse 2.7e-8）。
**编译门实证**：head_dim>128 在 BLOCK_M=64 触发 192KB UB overflow（requires 1587200 vs 1572864 bits，自动多缓冲
关闭仍溢出），BLOCK_M=32 逃逸确认——已建议回流 backend_facts。
未做：③simulator 瓶颈定型、P4 双盲回放、P5 KB 入库（本卡为 workflows 源卡，未改 source_family=ops 入 KB）。
边界登记：head_dim>256（512/768/1024）需 D-split（对齐 CANN D-workspace），未实现未验证。