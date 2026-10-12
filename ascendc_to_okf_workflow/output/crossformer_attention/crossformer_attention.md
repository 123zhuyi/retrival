---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: CrossformerAttention（分组/窗口化多头注意力）Ascend C → Triton 语义转化卡
description: CrossFormer 窗口化多头自注意力——把 token 序列（可选 reshape 为 H×W 空间）按 (gh,gw) 窗口分组，组内做 QKᵀ·scale + 加性 pos_bias/mask → softmax → ·V。CV 融合（Cube matmul + Vector softmax），映射为标准批多头 flash 注意力（在线 softmax 复刻）. 关键决策点：窗口分组是 host 侧布局、flash 在线 softmax 复刻、V 定向、尾列 -inf。
tags: [attention, cv_fusion, matmul, flash_attention, softmax, grouped, windowed, crossformer, multi-head]
confidence: inferred
status: draft
resource: zmm_source_retrieval/src/flash_attention_score（cann/ops-transformer @ d28d8f99a）+ src/flash_attention_score_grad 工作流 output/crossformer_attention
created_at: "2026-09-21"
updated_at: "2026-09-21"
---

sources:
  - url: zmm_source_retrieval/src/flash_attention_score/level4_target/CrossformerAttention/definition.json
    role: primary

# CrossformerAttention（分组/窗口化多头注意力）

**总述**：对 token 序列做（可选窗口化的）多头 scaled dot-product 注意力。语义 = level4 `definition.json` 的
torch golden：`qh·khᵀ·scale(+pos_bias)(+mask) → softmax(-1) → ·vh`；窗口分组（feature_shape + group_size 的
空间 tiles）是 **host 侧 layout 变换**，kernel 本体退化为**标准批多头 flash 注意力**。
**kernel 类型：CV 融合**（Cube matmul ×2 + Vector softmax/exp/max）。映射对象的 CANN 参考是
`flash_attention_score` 的 pse_shift/atten_mask/scale_value 语义路径（`docs/aclnnFlashAttentionScore.md:23`）
与 FlashSoftmaxV2 在线重构算法（`docs/FA算子设计介绍.md` §1 步 1~4）；窗口分组的算法结构取 CrossFormer/
level4，与量化版 swin 同构但非量化。

## 1. 功能与公式（详设 §1.1）

- 功能：窗口内多头自注意力，含加性位置偏置与掩码。
- 计算公式（level4 reference 反推，flash 在线软复制品）：
  $$
  \begin{aligned}
  &\text{score}[h,m,j] = \text{scale}\cdot\sum_d q[m,h,d]\cdot k[j,h,d] + \text{pos\_bias}[h,m,j] + \text{mask}[h,m,j]\\
  &P[h,m,j] = \text{softmax}_j(\text{score}[h,m,\cdot]) = e^{(score - \max_j)}/\textstyle\sum_j e^{(score-\max_j)}\\
  &\text{out}[m,h,d] = \sum_j P[h,m,j]\cdot v[j,h,d]
  \end{aligned}
  $$
  其中 scale = 属性或 `head_dim^{-0.5}`，head_dim = C/n_heads；窗口分组把 N(=H·W) token 分成
  N_w=(H/gh)·(W/gw) 个 G=gh·gw 窗口，窗口内 token 按本地索引 (lh,lw)→lh·gw+lw 排布。
- **语义分层**：
  - kernel（Triton）：纯批注意力，输入为 (N_w·G, C) 窗口连续布局。
  - host/layout：窗口分组 `idx`/逆排列 `inv`（wrapper 内 torch gather）——**窗口重排不在 kernel**。
  - benchmark：level4 `definition.json` reference（torch）为**单标杆**；公式超集项——mask/pos_bias 的
    窗口局部切片应用（见 §2 约束 3）。
- 计算流程概述：分组(Q/K/V) → 分头 → flash：QKᵀ(dot)→×scale→+bias→在线softmax(exp/max/α补偿)→P·V(dot)→除 l → 逆排列写回。
- 锚点：`src/flash_attention_score/docs/FA算子设计介绍.md:1-19`（§1 计算过程，flash 4 步）；
  `docs/aclnnFlashAttentionScore.md:23`（公式）；`definition.json reference`（golden）。

## 2. 原型与约束（详设 §2 / §2.1）

### 输入输出（来源：level4 `definition.json`）

| 参数名 | 入参类型 | 必选性 | dtype | format | shape | 语义 |
|---|---|---|---|---|---|---|
| x | 输入 | 必选 | fp16/bf16/fp32 | ND | (M,C)/(B,M,C) | token 序列 / 空间展平(self q/k/v) |
| q/k/v | 输入 | 可选 | 同 x | ND | (M,C) | 独立 q/k/v，缺省=x |
| group_size | 标量 | 必选 | int/[int,int] | — | (gh,gw) | 窗口大小 |
| num_heads | 标量 | 可选 | int | — | — | 头数，缺省 1 |
| mask | 输入 | 可选 | fp16/fp32 | ND | (H,N,N)/(N,N) 或 (H,G,G) | 加性掩码（窗口局部） |
| pos_bias | 输入 | 可选 | fp16/fp32 | ND | 同上 | 加性位置偏置（窗口局部） |
| scale | 标量 | 可选 | float | — | — | 缩放，缺省 head_dim^-0.5 |
| feature_shape | 标量 | 可选 | [H,W] | — | — | x 空间形状用于分组前 reshape |
| output | 输出 | — | 同 x | ND | (M,C) | 注意力输出 |

### 原型约束

- head_dim=C//num_heads 须整除（torch `.view` 隐含），CANN 对应 head_num（`flash_attention_score_def.cpp:161`）。
- 窗口分组要求 (H%gh)==0、(W%gw)==0（reference 直接 view/permute 不 pad）。
- ⚠️ **level4 reference 缺陷**：feature 分支把全序列 mask/pos_bias 误用于窗口 attention，shape 不广播会报错。
  本卡契约：mask/pos_bias 按**窗口局部坐标** (H_heads, G, G) 应用（feature case 实际给出即窗口局部尺寸），
  与 reference 在 'feature 无 bias' 与 '单窗口带 bias' 两个干净路径严格一致；'feature+大 mask' 组合以本卡契约为准。
- 非 feature 时 group_size 被 reference 忽略（走全序列单窗口 MHA）。
- 拦截校验：无（基准语义由 README/使用方给定）。

## 3. 规格约束（详设 §3）

| 类别 | 取值 | 来源锚点 |
|---|---|---|
| 支持 NPU 型号 | Ascend910B/A2、950 等（CANN 9.1.0） | hardware_platforms.md |
| dtype 全集 | fp16/bf16/fp32（中间 attn/o fp32 累加） | level4 workload / def.cpp |
| format | ND | level4 workload |
| 确定性计算 | 是（单 window/head 独立，无跨核规约/原子） | 骨架设计 |
| 交付形式 | python 直调（Triton） | 工作流 |

### 架构差异

- FAS 参考分 arch32（910B，KERNEL_TYPE_MIX_AIC_1_2）/arch35（950 regbase）两套流水；level4 语义与
  Triton 骨架不依赖具体 arch，映射取算法结构（flash 重构），不照搬规格/slot 时序。

## 4. 分核策略（详设 §4.1.1 → Triton grid）

- Ascend C 事实：FAS `tiling.cpp` 任务轴 B×H×S2，usedCoreNum 分核，Skv 循环进核内。
- **Triton 决策**：
  - 任务轴 = (N_w · n_heads · cdiv(G, BLOCK_M)) —— query 分块做任务（输出轴），key 分块进 flash 外循环。
  - 任务块总量公式：`NUM_BLOCKS = N_w · n_heads · cdiv(G, BLOCK_M)`。
  - 核分配范式：persistent-core(grid-stride 交织)，`for task in range(pid, NUM_BLOCKS, NUM_CORE)`；
    `grid = min(NUM_BLOCKS, num_core)`（CV 融合取 Cube 核数，C-011/operator_class_patterns §0）。

## 5. BLOCK / Tiling 参数（详设 §4.1.2 / §4.1.3）

### 映射

| tiling 字段 | 语义 | Triton 对应 | 候选值 |
|---|---|---|---|
| S1（key 分块） | 在线 softmax 的 key 步长 | BLOCK_N | {64,128,256}，基线 128 |
| S2（query 任务） | 任务内 query 分块 | BLOCK_M | {32,64,128}，基线 64 |
| head_dim（非 2 幂如 96） | feature 维 | BLOCK_D 上浮 2 幂 + `d<HEAD_DIM` mask | 128 |

BLOCK_M/BLOCK_N 固定；大 G（>BLOCK）走 flash 多块，`G≤BLOCK_N` 时逻辑仍自动正确（尾列 -inf）。

## 6. 片上内存预算（详设 §5）

| buffer | Triton tile | 字节 | 校核（Ascend910B A2 192KB） |
|---|---|---|---|
| Q | [BLOCK_M, BLOCK_D] fp16 | 64×128×2=16KB | ✅ |
| K / V 分块 | [BLOCK_D,BLOCK_N]×2 fp16 | 2×16KB=32KB | ✅ |
| attn s | [BLOCK_M,BLOCK_N] fp32 | 64×128×4=32KB | ✅ |
| o / l / m | 累加器 | ≈16KB | ✅ |
| **合计** | | ≈97KB ≈ 51% | ✅ 富余；fp32 输入 ×4 自动降档 |

- double buffer：Triton 由 multibuffer/num_stages 控制，但 C-006 运行时在 Triton-Ascend 不生效——
  标记 ⚠️ 性能两态，不写实数值。
- 大 G 的 flash 循环每迭代仅一块 K/V，UB 不随 G 增长。

## 7. 流水编排与 Triton 可达性（详设 §6）

### 流水结构

flash 外循环（key 分块）内：load K/Vᵀ → dot(q,kᵀ)×scale(+bias) → 在线 softmax（max/exp/α 补偿/l 更新）→
dot(p·v) 累加 o。单 kernel 完成 Q 块，无跨块规约。

### 可达性判定表

| level4/CANN 手段 | 锚点 | 可达性 | Triton 写法 / 降级 | 预期影响 |
|---|---|---|---|---|
| FlashSoftmaxV2 在线重构 | FA设计介绍.md §1 步1~4 | ⚠️ 部分（正确性补足） | tl.dot + loop-carried tl.maximum/exp/sum + α=exp(m_old−m_new) | 数学等价；向量化由编译器 |
| key 尾列遮蔽（mask → −inf） | atten_mask 布尔 select | ✅ | `s = tl.where(n_ok, s, -inf)`（**必须**，C-031） | 防假质量 |
| 加性 pos_bias/mask | FAS pse/atten_mask | ✅ | bias tensor `(n_heads,G,G)` 全窗共享，s+=bias | |
| double buffer 掩盖 | — | ⚠️ 编译器自动 | num_stages/multibuffer（C-006 运行时无效） | 性能两态 |
| 窗口分组重排（host） | — | ✅（不在 kernel） | wrapper `idx`/`inv` gather | 一次 host 数据移动 |
| Reg gather/scatter（NZ/Transpose） | swin 量化路径 | ❌ | 不做（level4 无关） | — |
| 原生在线 softmax 指令 | — | ❌ | 显式循环复刻（数学等价） | — |

## 8. Triton 骨架（生成用伪码，P2.5 验证过）

```python
# 可运行实现见 output/crossformer_attention/triton_skeleton.py（P2.5 全 PASS）。要点：
@triton.jit
def _crossformer_attn_kernel(q,k,v,bias,out, NW,G,C,HEAD_DIM,N_HEADS,scale,STRIDE_T,
                             HAS_BIAS,IS_FP32,OUTPUT_DTYPE: constexpr, BLOCK_M,BLOCK_N,BLOCK_D):
    pid = tl.program_id(0)
    num_m = (G + BLOCK_M - 1)//BLOCK_M; NTOT = NW*N_HEADS*num_m
    dcol = tl.arange(0,BLOCK_D); d_mask = dcol < HEAD_DIM
    for task in range(pid, NTOT, tl.num_programs(0)):      # persistent grid-stride
        blk=task%num_m; rem=task//num_m; head=rem%N_HEADS; w=rem//N_HEADS
        w0=w*G; feat=head*HEAD_DIM; foff=feat+dcol
        mrow=blk*BLOCK_M+tl.arange(0,BLOCK_M); m_ok=mrow<G; row=w0+mrow
        qoff = q + row[:,None]*STRIDE_T + foff[None,:]
        q = tl.load(qoff, mask=m_ok[:,None]&d_mask[None,:], other=0.0)
        # 累加器初值用全 False mask load 产零/产 -inf（C-016 禁 tl.zeros）
        o_i = tl.load(qoff, mask=(mrow[:,None]>G), other=0.0).to(tl.float32)
        rowv = w0+tl.arange(0,BLOCK_M)
        l_i = tl.load(q+rowv, mask=(mrow>G), other=0.0).to(tl.float32)
        m_i = tl.load(q+rowv, mask=(mrow>G), other=float("-inf")).to(tl.float32)
        for n0 in range(0, G, BLOCK_N):
            ncol=n0+tl.arange(0,BLOCK_N); n_ok=ncol<G; nrow=w0+ncol
            kt = tl.load(k+nrow[None,:]*STRIDE_T+foff[:,None], mask=n_ok[None,:]&d_mask[:,None], other=0.0)
            v  = tl.load(v+nrow[:,None]*STRIDE_T+foff[None,:], mask=n_ok[:,None]&d_mask[:,None], other=0.0)  # (BLOCK_N,BLOCK_D)
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
# wrapper：feature 时 torch 分组 idx + 逆排列 inv（k ernel 输出分组布局 → o[inv]）
```

wrapper 要点：grid=min(NW·n_heads·num_m, cube_core)；feature 分组/逆排列在 host；scale runtime 标量（C-013）；
IS_FP32 constexpr（fp32→ieee dot）；HAS_BIAS constexpr；bias 合并 pse+mask。

## 9. 精度标杆与验收（详设 §测试设计）

- 标杆类别：单标杆 = level4 `definition.json reference`（torch fp32 重算 softmax）。
- 标杆实现要点：`dh=to_heads; attn=matmul(qh.float(),khᵀ)·?;+=bias;softmax(-1);@vh.float()`（golden_run）。
- **算子级验收标准**：对 golden 的 `mae/rmse ≤ 2e-2`（f16/bf16 输入）或 `≤1e-3`（fp32 输入）；形状一致。
  P2.5 实测达 f16 1.3e-4 / bf16 4.6e-6 / fp32 3.1e-8 量级。
- **benchmark 语义超集**：mask/pos_bias 的窗口局部切片应用是本卡契约扩展（reference 缺陷）；其余参数
  （scale、q/k/v、feature_shape）均在验收域。workload 50 case 覆盖 f16/bf16/fp32、单窗/多窗、mask/pos_bias。
- 典型用例：`(M=196..7921, C=96..1024, n_heads=1..512, G≤7921)`。
- 边界：G<BLOCK_N 尾块（-inf）、head_dim 非 2 幂（BLOCK_D mask）、fp32 ieee、非对齐小窗。

## 10. Agent actions（生成指引）

1. 识别：多窗口/分组多头注意力，CV 融合（Cube 两 matmul + Vector softmax）→ 映射**标准批多头 flash 注意力**。
2. 切分轴取 (N_w·n_heads·num_m)，key 分块进 flash 外循环；persistent grid-stride（勿「一块一 program」）。
3. BLOCK 候选取 BLOCK_M∈{32,64}、BLOCK_N∈{64,128}、BLOCK_D=上浮 2 幂≤128；UB 预算见 §6（fp16 富余，fp32 降档）。
4. 精度提升点：attn s 与 o 累加保留 **fp32**；fp32 输入 dot 用 `input_precision="ieee"`（C-029：f16/bf16 不写 precision，禁写 tf32）。
5. **不可达/必做**：窗口分组**在 wrapper 做**（host gather idx/inv），kernel 保持连续仿射 load（勿 Reg gather）；
   flash 在线 softmax 手动维护 running max/α 补偿（无原生指令，勿跳过）；**key 尾列必须 `-inf`（C-031）**。
6. **勿复刻**：FAS 的 S1S2 时隙/QK 双 buffer 等为硬件手段，Triton 单 kernel 已完成；swin 量化 NZ/Transpose 通路无关不做。
7. `p@V` 的 V 加载方向须 `(BLOCK_N, BLOCK_D)`（key×feature）——勿误用 Kᵀ 的 (D,N)（C-030）。
8. 验收按 §9 标准（对 f16/bf16 用 2e-2 容差，fp32 用 1e-3）。

## 11. 复现性声明

本卡由 level4 golden + CANN FAS 结构转化（confidence: inferred）。P2.5 **①编译门 + ②正确性门**已在真机
Ascend910 + CANN 9.1.0 + torch_npu 2.10 跑通（7 组代表形状对 golden 全 PASS，见 mapping §P2.5/validation_report）。
未经硬件实测：③simulator 瓶颈定型、③.5 board 选型、④IR 取证未跑（P2.5 降级为 ①②）；bias 组合 case 以
窗口局部契约为准（reference 缺陷）。回放验证：**未做（P4 双盲回放未执行）**——frontmatter confidence 保持 inferred。