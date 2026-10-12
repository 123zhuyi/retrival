---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: flash_attention_score（SDPA 稠密 FlashAttention）Ascend C → Triton 语义转化卡（v2）
description: level4 SelfAttention 的注意力核——稠密多头 scaled-dot-product 注意力 softmax(QKᵀ·scale)·V，CV 融合，映射标准批多头 flash（在线 softmax 复刻）。v2 核心：service level4 接口差（4 段投影归 wrapper）+ BLOCK 几何绑定 CANN host 档位阶梯（D-tier）+ 实证 C-032 在 flash 上呈「静默降精度」形态（非编译硬失败），D-tier 降 BLOCK_M 逃逸。
tags: [attention, cv_fusion, matmul, flash_attention, softmax, multi-head, self_attention, host_tiling_ladder, sdpa]
confidence: inferred
status: draft
resource: zmm_source_retrieval/src/flash_attention_score_v2（cann/ops-transformer @ d28d8f99a re-extract v2）+ 工作流 output/flash_attention_score_v2
created_at: "2026-10-10"
updated_at: "2026-10-10"
---

sources:
  - url: zmm_source_retrieval/src/flash_attention_score_v2/level4_target/SelfAttention/definition.json
    role: primary
  # 说明：本卡为 workflows 源卡（双副本 output/flash_attention_score_v2/ 为源）。KB 版如需入库须
  # sources[].url=GitCode @sha 永久链接、role=example；source_family=ops 已就位。

# flash_attention_score（多头稠密注意力）v2

**总述**：对 token 序列做稠密多头 scaled-dot-product 注意力。服务目标 = level4 `SelfAttention` 的 torch golden：
`q_proj/k_proj/v_proj(x) → scores = matmul(q.float(), k.float().T)/√hd → softmax(-1) → matmul(P, v.float()) → out_proj`。
**kernel 类型：CV 融合**（Cube matmul ×2 + Vector softmax）。**接口差（v2 §0 显式拆解）**：golden 在注意力前内建
4 段 `nn.Linear` 投影（seed=42）且 out 再 `out_proj`；CANN `flash_attention_score` kernel **只覆盖注意力核**
`softmax(QKᵀ·scale)·V`——投影/out_proj 归 **wrapper torch**，head 重排（`view(H,hd)`+`transpose(1,2)`）在 wrapper 完成。
与 crossformer_v2 同构移植：**NW=batch、G=S、head 特征切片**，但本 path 走**朴素稠密**主纲（无 pse/atten_mask/
sparse/NZ/fp8/rope 全不触发，pse_type 合规即可不写 bias 分支）。

## 1. 功能与公式（详设 §1.1）

- 功能：稠密多头自注意力（query_length == key_length == S），含在线 softmax 数值稳定化。
- 公式（level4 reference 反推，flash 在线 softmax）：
  $$
  score[m,j] = scale\cdot\sum_d q[m,h,d]\cdot k[j,h,d],\quad scale=head\_dim^{-0.5}
  $$
  $$
  P[m,j] = softmax_j(score[m,\cdot]) = e^{(score-\max)}/\textstyle\sum_j e^{(score-\max)}
  $$
  $$
  out[m,h,d] = \sum_j P[m,j]\cdot v[j,h,d]
  $$
  head_dim = D/n_heads（D=特征维）。每 (batch, head) 对自身 G 长序列独立 flash。
- **语义分层**：kernel=纯注意力（输入投影后的 q/k/v，token 行主序 (B·S,C)，C=N_HEADS·HEAD_DIM）；
  host/wrapper=4 段 `nn.Linear` 投影 + head 切片（`view(H,hd).transpose(1,2).contiguous().view(B·S,C)`）；benchmark=level4
  `definition.json reference` 单标杆。
- 计算流程：投影 q/k/v → flash 外循环（key 分块：dot(q,kᵀ)·scale → 在线软 max/α 补偿/l 更新 → dot(p,v) 累加）→
  ÷softmax_sum → 写回 → out_proj。
- 锚点：`docs/FA算子设计介绍.md:1-19`（flash 4 步）；level4 `definition.json` reference。

## 2. 原型与约束（详设 §2 / §2.1）

### 输入输出（来源：`op_host/flash_attention_score_def.cpp`）

| 参数名 | 入参类型 | 必选性 | dtype | format | shape | 语义 |
|---|---|---|---|---|---|---|
| query | 输入 | 必选 | fp16/bf16/fp32 | ND | (S,H,hd)/(B,H,S,hd)… | 投影后 Q（level4 内建 q_proj） |
| key | 输入 | 必选 | fp16/bf16/fp32 | ND | 同 query | 投影后 K |
| value | 输入 | 必选 | fp16/bf16/fp32 | ND | 同 query | 投影后 V |
| real_shift | 输入 | 可选 | — | ND | — | pse 移位（level4 不触发） |
| drop_mask | 输入 | 可选 | uint8/bool | ND | — | dropout mask（不触发） |
| padding_mask / atten_mask | 输入 | 可选 | uint8/bool | ND | — | 遮蔽（不触发） |
| prefix / actual_seq_qlen / actual_seq_kvlen | 输入 | 可选 | int64 | ND | — | PND/变长（不触发） |
| q_start_idx / kv_start_idx | 输入 | 可选 | int64 | ND | — | 分块起点（不触发） |
| d_scale_q/k/v / p_scale / sink | 输入 | 可选 | fp32 | ND | — | fp8/delta（不触发） |
| query_rope / key_rope | 输入 | 可选 | — | ND | — | 旋转位置（不触发） |
| softmax_max / softmax_sum / softmax_out | 输出 | — | fp32 | ND | (B,N,S,8) | 三路中间值（**level4 ❌→不需要**） |
| attention_out | 输出 | 必选 | 同 x | ND | (S,D) | 注意力输出 |

### 属性（def.cpp:157-169）

| 属性 | 类型 | 默认值 | 语义 |
|---|---|---|---|
| scale_value | float | 1.0 | 缩放（level4 用 head_dim^-0.5） |
| keep_prob | float | 1.0 | dropout 保留率 |
| pre_tockens / next_tockens | int | INT_MAX | 稀疏窗口/因果跨度 |
| head_num | int | 必选 | 头数 |
| input_layout | string | 必选 | BSH/SBH/BNSD/TND 等布局归一 |
| inner_precise | int | 0 | 精度档（0=精确，高性能够档不选） |
| sparse_mode | int | 0 | 稀疏模式（本 target 0=稠密） |
| pse_type | int | 1 | 位置编码类型（合法性检查位） |
| seed / offset | int | 0 | dropout 随机 |
| out_dtype | int | 0 | 输出 dtype |
| softmax_out_layout | string | "" | 三中间输出布局 |

### 原型约束

- head_dim=D/n_heads 须整除（torch `.view` 隐含；`def.cpp:161 head_num`）。
- level4 自注意力 query_length==key_length==S；无 out 维。CANN pse_type 合法性使 pse 分支可合并于统一语义。
- **head_dim 契约（v2）**：C=64..8192、n_heads 可大 → head_dim∈{64,128,256}（level4 实际）。Triton `BLOCK_D ≥ head_dim`
  须 2 幂上浮；大 D 的 UB 约束见 §6（**C-032**）。

## 3. 规格约束（详设 §3）

| 类别 | 取值 | 来源锚点 |
|---|---|---|
| 支持 NPU 型号 | Ascend910B/A2（arch32 MIX_AIC_1_2）、9372、950（CANN 9.1.0） | hardware_platforms.md |
| dtype 全集 | fp16/bf16/fp32（中间 attn/o 保 fp32 累加） | def.cpp / level4 workload |
| format | ND（投影后 q/k/v token 行主序） | def.cpp FORMAT_ND |
| 确定性计算 | 是（单 window/head 独立，无跨核规约） | 骨架设计 |
| 交付形式 | python 直调（Triton）；投影=wrapper torch | 工作流 |

### 架构差异
FAS 参考 arch32（910B KERNEL_TYPE_MIX_AIC_1_2，S1S2_bn2gs1 六阶段 CV 换手）/ arch35（DAV_3510 regbase，910b
生产路径）两套。level4 与 Triton 骨架取算法结构（flash 重构），不照搬 slot 时序。**regbase 深流水（taskId ring/
L1 3-buffer）不可移植**（v1.5 L2 排除清单）。CANN 六模板分派（TNDSameAB/S1S2/S1/SameAB…）在 level4 命中 S1S2 模板
（S1==S2==S），sv 循环种类进核内（`s1s2_bn2gs1.h:773-778`）。

## 4. 分核策略（详设 §4.1.1 → Triton grid）

- Ascend C：FAS `tiling` 任务轴 B×H×S2，usedCoreNum 分核，KV(S2) 循环进核内、不进多核（
  `s1s2_bn2gs1.h:773,861`）。
- **Triton 决策**：
  - 任务轴 = (B · n_heads · cdiv(S, BLOCK_M))（query 输出轴做任务，key 分块进 flash 外循环）。
  - `NUM_BLOCKS = B·n_heads·cdiv(S,BLOCK_M)`；persistent-core grid-stride：
    `grid = min(NUM_BLOCKS, num_core)`，核内 `for task in range(pid, NUM_BLOCKS, NUM_CORE)`（CV 取 Cube 核数 C-011）。

## 5. BLOCK / Tiling 参数（详设 §4.1.2 / §4.1.3）—— v2 绑定 CANN host 档位阶梯

### v2 主证据：arch35 host 档位阶梯（v1.5 第一顺位）

CANN `op_host/arch35/flash_attention_score_tiling_basic.cpp`：
- **D 轴分段 `CalcDBasicBlock`(:51)**：`dBasicBlock = AlignUp(dSize,64)` → {64,128,192,256}；>256 → 768。
  → **D-tier = {64,128,192,256,768}**，Triton 化 = `BLOCK_D = ≥head_dim 最小 2 幂`。
- **S1/S2 主档 `CalcS1S2BasicBlock`(:79)**：缺省 128/128；D>256 → fp32?64/128:128/128；DN 优化 128/256。
  Level4 主流（head_dim≤256）命中缺省 → CANN 默认档 = **128/128**。
- **关键现实（v1.5 L1-unlock/L2-bound）**：CANN 128/128 靠 **L1 3-buffer + D-workspace** 撑起，Triton 无此机制 →
  128/128 复刻会 192KB UB 溢出。故 Triton 取 **64/128 可移植近邻**（CANN 大块 S2=1024 不可照抄，MPA 启动次数另权衡）。

### 映射表

| tiling 字段 | CANN 档位（锚点） | Triton 对应 | 候选值 |
|---|---|---|---|
| D（head_dim） | D-tier AlignUp(D,64)→{64,128,192,256}/768 | BLOCK_D=≥head_dim 最小 2 幂 | {128,256,512} + d<HEAD_DIM mask |
| S1（key 分块） | 默认 128 | BLOCK_N | 128（head_dim≤128）；64（大 D 逃逸）|
| S2（query 任务） | 默认 128（任务轴） | BLOCK_M | 64（head_dim≤128）；32（大 D 逃逸）|

**BLOCK_D>128 自动降 BLOCK_M=32/BLOCK_N=64**（fp32 O 累加器主导 UB，见 §6）。head_dim>256 需 D-split（登记边界）。

**精度最简方案（v2 补，kernel Compute 累加结构）**：CANN kernel arch35 bmm2 对 `p×V` 是**原生单 dot、fp32 累加**（`acc + dot(p.to(fp16), v)`，无 hi/lo 低比特拆分补偿）。triton 侧默认单 dot 即可命中硬件原生精度；**仅当 verify 参考容差严于硬件原生水平（half atol<~1e-2 / 早期模板的 1e-1 档）时才考虑 hi/lo 2-dot 补偿**。op_8 实测（910b + aclnn 宽松容差 atol 9e-2）单 dot 全量 50/50 过、+13.1%——**先试单 dot，别默认上补偿**（见方法论 §D：精度补偿/中间重算是"尝试过的轴无效"也扫不到的隐藏开销）。

## 6. 片上内存预算（详设 §5）—— v2 修正 + C-032 静默衰减实证

flash 骨架 flash 累加区（fp32 O 累加器是主导项，BLOCK_N 影响甚微）：
| 配置 | buffer | 字节 | 校核（192KB）|
|---|---|---|---|
| head_dim≤128（primary）| q16K + k/v 各 16K + s/p 各 32K + o_i 32K | ≈112KB | ✅ ≈60% |
| head_dim=256（BM32/BN64/D256）| q16K + k32K + v32K + s/p 各 8K + o_i 32K | ≈128KB | ✅（BLOCK_M=32）|
| head_dim=256（BM64/BN128/D256）| o_i=64K + s32K + p32K + … | **>192KB** | ❌ **UB 越界** |

**v2 实证（flash 特有，异于 crossformer 编译硬失败）**：head_dim=256 在 BLOCK_M=64 时 **不报 `ub overflow` 错误**，
而是**静默降精度**——初跑 rmse 9.4e-3（flash 的在线 softmax + O 累加器拼装让溢出被 compiler code-motion/multibuffer
兜「通」，而非交叉软 max 等路径；比 crossformer 的显式 `requires 1587200 vs 1572864 bits` **更隐蔽**）。**D-tier 降
BLOCK_M=32/BLOCK_N=64 后 rmse→3.15e-5（300× 回归）**。

**结论**：大 D（BLOCK_D>128）模型 = fp32 累加器放不下 → canonical 近邻（64/128）覆盖 head_dim≤128；更大 head_dim 走
BLOCK_M=32 或 D-split（对齐 CANN D-workspace）。**对写卡者的判别**：`hd>128 + BLOCK_M=64` 出现高 rmse ≠ 数值/舍入问题，
**先查 UB 溢出**（flash/CV 融合算子可能静默而非报错）。

## 7. 流水编排与 Triton 可达性（详设 §6）

flash 外循环（key 分块）内：load Kᵀ/V → dot(q,kᵀ)·scale → 在线 softmax（max/exp/α 补偿/l 更新）→ dot(p·v) 累加 o。
单 kernel 完成 Q 块，无跨块规约。

| level4/CANN 手段 | 锚点 | 可达性 | Triton 写法 / 降级 |
|---|---|---|---|
| FlashSoftmaxV2 在线重构 | FA设计介绍.md §1 | ⚠️ 部分（正确性补足）| tl.dot + loop-carried max/exp/sum + α=exp(m_old−m_new) |
| key 尾列遮蔽→-inf | 稠密边界 | ✅ | `s = tl.where(n_ok, s, -inf)`（**必须**，C-031）|
| 4 布局归一（BSH/SBH/BNSD/TND）| tilingkey Layout 位 | ✅ | wrapper 读 tensor.stride() 归一化，kernel 仅 stride 仿射，不入分支 |
| fp32 中间域 / 内精度 | inner_precise | ✅ | q/k/v、scores、P、o 保 fp32；fp32 输入 dot `input_precision="ieee"`（C-029）|
| 大 D D-workspace（bmm2 落 GM）| arch35 GetWorkspaceSize | ❌ | 需自建 D-split（登记边界）|
| regbase 深流水（L1 3-buffer/taskId）| arch35 `_tiling_basic.cpp` | ❌ | 排除清单（v1.5 L2）|
| softmax_max/sum + ×8 补位输出 | def.cpp:137-156 | ❌→不需要 | level4 golden 内部重算 softmax、只取 out；省 3 路 GM 写 |
| bmm1 输出经 workspace/GM | tiling_general | ❌→不需要 | tl.dot 板载融合，无 GM 中转 |

## 8. Triton 骨架（生成用伪码）—— P2.5 验证过

```python
# 可运行实现见 output/flash_attention_score_v2/triton_skeleton.py（P2.5 ①②门 7/7 PASS）。
# wrapper 先做 4 段投影（seed=42 nn.Linear）：q/k/v/out_proj；head 切片列区位偏移 w*G + head*HEAD_DIM。
@triton.jit
def _selfattn_kernel(q,k,v,out, NW,G,C,HEAD_DIM,N_HEADS,scale,STRIDE_T,
                     IS_FP32,OUTPUT_DTYPE:constexpr, BLOCK_M,BLOCK_N,BLOCK_D):
    pid = tl.program_id(0)
    num_m = (G + BLOCK_M - 1)//BLOCK_M; NTOT = NW*N_HEADS*num_m          # NW=batch, G=S
    dcol = tl.arange(0,BLOCK_D); d_mask = dcol < HEAD_DIM
    for task in range(pid, NTOT, tl.num_programs(0)):                    # persistent grid-stride
        blk=task%num_m; rem=task//num_m; head=rem%N_HEADS; w=rem//N_HEADS
        w0=w*G; feat=head*HEAD_DIM; foff=feat+dcol
        mrow=blk*BLOCK_M+tl.arange(0,BLOCK_M); m_ok=mrow<G; row=w0+mrow
        qoff = q + row[:,None]*STRIDE_T + foff[None,:]
        q = tl.load(qoff, mask=m_ok[:,None]&d_mask[None,:], other=0.0)
        o_i = tl.load(qoff, mask=(mrow[:,None]>G), other=0.0).to(tl.float32)   # C-016
        rowv = w0+tl.arange(0,BLOCK_M)
        l_i = tl.load(q+rowv, mask=(mrow>G), other=0.0).to(tl.float32)
        m_i = tl.load(q+rowv, mask=(mrow>G), other=float("-inf")).to(tl.float32)
        for n0 in range(0, G, BLOCK_N):
            ncol=n0+tl.arange(0,BLOCK_N); n_ok=ncol<G; nrow=w0+ncol
            kt = tl.load(k+nrow[None,:]*STRIDE_T+foff[:,None], mask=n_ok[None,:]&d_mask[:,None], other=0.0)   # (BLOCK_D,BLOCK_N) 镜像转置
            v  = tl.load(v+nrow[:,None]*STRIDE_T+foff[None,:], mask=n_ok[:,None]&d_mask[:,None], other=0.0)     # (BLOCK_N,BLOCK_D) C-030
            s = _dot(q, kt, IS_FP32)*scale                                       # fp32 input → ieee (C-029)
            s = tl.where(n_ok[None,:], s, float("-inf"))                          # C-031 尾列 -inf
            m_c=tl.max(s,1); m_n=tl.maximum(m_i,m_c); alpha=tl.exp((m_i-m_n).to(tl.float32))
            p=tl.exp(s-m_n[:,None]); l_i=l_i*alpha+tl.sum(p,1)
            o_i=o_i*alpha[:,None]+_dot(p.to(OUTPUT_DTYPE), v, IS_FP32); m_i=m_n
        out_t=(o_i/tl.maximum(l_i,1e-24)[:,None]).to(OUTPUT_DTYPE)
        tl.store(out+row[:,None]*STRIDE_T+foff[None,:], out_t, mask=m_ok[:,None]&d_mask[None,:])
# wrapper：bd = next_pow2(head_dim); BM,BN = (64,128) if bd<=128 else (32,64)（C-032 D-tier）;
#          grid = min(B*n_heads*num_m, core); pyrt后 return (out @ Wo)（out_proj torch）
```

## 9. 精度标杆与验收（详设 §测试设计）

- 标杆类别：单标杆 = level4 `definition.json reference`（torch fp32 重算 softmax，含 4 段 seed42 投影）。
- 验收标准：对 golden `mae/rmse ≤ 2e-2`（f16/bf16）或 `≤1e-3`（fp32）；形状一致、dtype 回写一致。
  P2.5 实测 f16 3.9e-6 / bf16 1.6e-5 / fp32 6.0e-9 / 大 S4096 flash 多块 1.4e-6 / hd256(BM32) 3.15e-5 / smallS 1.1e-5。
- benchmark 语义超集：本 target 仅 x + n_heads，无源码外参数；pse/mask/drop/sparse/fp8/rope 均为 FAS 可选面但
  **本卡不触发**（映射判定存「可达不触发」）。
- 典型用例：`(B=1..2, S=64..4096, D=512..8192, n_heads=8..32, hd=64..256)`。
- 边界：S 尾块 -inf（C-031）、head_dim 非 2 幂（BLOCK_D mask）、fp32 ieee（C-029）、head_dim=256 走 BM32（§6）。

## 10. Agent actions（生成指引）

1. 识别：稠密多头 SDPA，CV 融合（Cube 两 matmul + Vector softmax）→ **标准批多头 flash 注意力**；接口差见 §1（投影归 wrapper）。
2. 切分轴 (B·n_heads·num_m)，key 分块进 flash 外循环；persistent grid-stride（勿「一块一 program」）。
3. **BLOCK 依 CANN host 档位阶梯（v2 核心）**：`BLOCK_D = next_pow2(head_dim)`（D-tier 缩放，禁硬编码 128）；head_dim≤128 →
   BLOCK_M=64/BLOCK_N=128；head_dim 128~256 → **BLOCK_M=32/N64**（**C-032**：BM64 会**静默降精度**非报错，先查 UB）；
   head_dim>256 → **D-split 登记未实现**。
4. 精度：attn s 与 o 累加保 **fp32**；fp32 输入 dot `input_precision="ieee"`（C-029，f16/bf16 不写，禁 tf32）；
   出写回输入 dtype（RTNE）。
5. **不可达/必做**：flash 在线 softmax 手动维护 running max/α 补偿（无原生）；**key 尾列必须 -inf（C-031）**；Kᵀ/V
   转置镜像跨步 load 勿 tl.trans（C-018）；V 按 (BLOCK_N,BLOCK_D) key×feature（C-030）；累加器勿 tl.zeros（C-016）。
6. **勿复刻**：FAS S1S2 时隙/QK 双 buffer/regbase 深流水/L1 3-buffer/D-workspace/softmax 三中间输出/workspace GM 中转——
   Triton 单 kernel 已完成/不需要（v1.5 L2）。CANN 128/128 默档与 1024 大块靠 L1/D-workspace，192KB 复刻溢出。
7. 验证：`f16/bf16 rmse≤2e-2，fp32≤1e-3`，大 S 须 flash 多块、多 batch 全覆盖；**hd>128 用例务必核对 rmse 非高**（UB 静默劣化指标）。
8. 验收按 §9 执行；投影=wrapper，勿将 head 重排写进 kernel（保持纯连续仿射 flash）。

## 11. 复现性声明

v2 卡由 level4 golden + CANN FAS 结构 + **arch35 host 档位阶梯**（v1.5 方法论）转化（confidence: inferred）。
P2.5 ①编译门+②正确性门已在真机 Ascend910（arch 9372）+ CANN 9.1.0 + torch_npu 2.10 跑通：7/7 PASS
（fp16/bf16/fp32 × hd64/128/256 × B1/B2 × 大 S4096 flash 多块，rmse 见 §9）。
**①门实证（C-032 flash 形态）**：head_dim=256 在 BLOCK_M=64 **静默降精度（rmse 9.4e-3），非编译硬失败**；
BLOCK_M=32/N64/D256 逃逸后 rmse→3.15e-5（300×）。与 crossformer_v2 的编译期 `requires 1587200 vs 1572864 bits`
（crossformer 硬失败）构成 C-032 双形态证据。
未做：③simulator 瓶颈定型（④降级）、P4 双盲回放、P5 KB 入库（本卡为 workflows 源卡）。
边界登记：head_dim>256（512/768/1024）需 D-split（对齐 CANN D-workspace），未实现未验证。