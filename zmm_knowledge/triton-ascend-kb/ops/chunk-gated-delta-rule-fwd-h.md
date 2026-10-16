---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: ops
title: ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡
description: GDN/KDA 共享的 chunk 间状态递推算子：v_new = u − w@h_prev，h_next = h_prev·exp(g_last)·exp2(gk_last) + kg^T@(v_new·decay)。CV 融合（MIX_AIC_1_2，Catlass 模板）+ 双流 ping-pong + 全 fp32 中间域。对 Triton 生成的关键决策：(b,hv,v_block) 状态条带常驻并行轴、V 维列条带独立性消解全部 GM workspace、禁 tl.trans 用转置寻址 load、g 的 chunk 内 cumsum 语义与 g_last 取 actual-1 行。
tags: [gdn, kda, gated_delta_rule, linear_attention, recurrent_state, cv_fusion, matmul, fp16, bf16, fp32_accum, varlen, chunk_gated_delta_rule_fwd_h]
confidence: inferred
status: draft
resource: https://github.com/jiangdm2024/flash-linear-attention-npu_dev/tree/4c78653f47beda77681aa725c7fa49c7ce8d86a9/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h
created_at: "2026-09-09T00:00:00Z"
updated_at: "2026-09-14T00:00:00Z"
---

sources:
  - url: https://github.com/jiangdm2024/flash-linear-attention-npu_dev/tree/4c78653f47beda77681aa725c7fa49c7ce8d86a9/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h
    role: primary
  - url: https://github.com/jiangdm2024/flash-linear-attention-npu_dev/blob/4c78653f47beda77681aa725c7fa49c7ce8d86a9/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/tests/pta/test_fwd_h.py
    role: example

> 迁移备注：本卡由 AscendC→OKF 转化工作流（zmm_source_retrieval/ascendc_to_okf_workflow，P0~P3 产出，过程产物在 `output/chunk_gated_delta_rule_fwd_h/`）适配入库，分析对象为 `jiangdm2024/flash-linear-attention-npu_dev` 仓（天津大学，非 CANN 官方仓）`fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h` 的本地快照（@4c78653f，分支 main，行号锚点以快照为准）。status: draft / confidence: inferred，未硬件实测；P4 双盲回放验证待执行。与 [RecurrentGatedDeltaRule（gated delta rule 单算例卡）](recurrent-gated-delta-rule.md) / [GDN 线性注意力三阶段优化案例](gdn-chunked-linear-attention.md) 分工：那两张是 gated delta rule 的通用范式/优化轨迹卡，本卡是 ChunkGatedDeltaRuleFwdH 算子（GDN/KDA 共享 chunk 间状态递推，CV 融合 + Catlass）的全规格语义转化卡——含 (b,hv,v_block) 状态条带并行化论证与跨核四段流水的 Triton 降级方案。

# ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）

逐 chunk 递推线性注意力的隐状态：`v_new = u − w@h_prev`；`h_next = h_prev·scalar_decay·key_decay + kg^T@(v_new·decay)`，输出每个 chunk 的起始状态 h（反向要用）、修正 v_new、可选 final_state。kernel 类型 **CV 融合**（`KERNEL_TASK_TYPE(*, KERNEL_TYPE_MIX_AIC_1_2)`，op_kernel/chunk_gated_delta_rule_fwd_h.cpp:119,123），基于 **Catlass 模板库**（`Catlass::Gemm::Kernel::GDNFwdHKernel`）。**依赖缺口必标注**：Catlass 的 Mmad block（`catlass/gemm/block/block_mmad*.hpp`）与 Reg VF 实现（`kernel_utils/vector/regbase.hpp`）**本体不在快照内**（manifest.md 缺省项 4），mmad 内部流水按黑盒处理；在仓内核心语义资产 = scheduler 任务/offsets 结构 + 两个 epilogue 的完整向量语义 + regbase VF 的数学定义 + golden 公式。

## 1. 功能与公式（详设 §1.1）

- 功能：GDN/KDA 前向的 chunk 间状态递推（h propagation），供后续 chunk 内注意力使用。
- 计算公式（对每个 batch n、v-head hv、chunk i；README.md:20-27 + golden tests/pta/test_fwd_h.py:111-115）：

$$\text{v\_new}[t] = u[t] - w[t]\, h_i, \qquad \text{decay}[t] = e^{\,g[\text{last}]-g[t]}$$
$$h_{i+1} = h_i \cdot e^{\,g[\text{last}]} \cdot 2^{\,g_k[\text{last},:]} \;+\; k_g^{\top}\big(\text{v\_new}\odot \text{decay}[:,None]\big)$$

  变量：$h_i \in \mathbb{R}^{K\times V}$（chunk i 起始状态）；$w,u$ 为 chunk 内 token 行（$[\text{BT},K]$/$[\text{BT},V]$）；$g\in$ 标量门（**自然对数域，chunk 内 cumsum 语义**，test_fwd_h.py:261 佐证）；$g_k\in$ 逐 K 门（**log2 域**，README.md:29-30 固定）；$g[\text{last}]$ = chunk 内最后一个有效 token 的值；$k_g$ = useGk 时由**调用方预衰减**的 k（`kg = k·2^{gk_last−gk}`，vnew epilogue 注释 block_epilogue_gdn_fwdh_vnew.hpp:407），否则 = k。末 chunk 的 $h_{i+1}$ 写入 final_state。
- **语义分层**：
  - GEIR kernel 本体：上式；三种 gate 组合（both / gk-only / g-only，kernel 分派 chunk_gated_delta_rule_fwd_h.cpp:49-60）；useExp2 模板恒 false，exp2 语义全部由 epilogue 的 ×ln2+Exp 实现（regbase.hpp:53-73）。
  - aclnn 接口层：① `state_v_first` 的 [K,V]↔[V,K] 由 TransposeLastTwo+ViewCopy 消化（aclnn_chunk_gated_delta_rule_fwd_h.cpp:83-94,321-366），kernel 只见 [K,V]；② `output_final_state=false` 时内部分配不可见 finalState（:330-336）；③ logical_* 7 属性由 l0op 从 shape 推导（op_api/chunk_gated_delta_rule_fwd_h.cpp:60-72）。
  - benchmark/测试层：golden `forward_h_trans_cpu` 全 fp32 递推（仅覆盖 g-only 路径），比较器 rel 1e-4 / 99.99% / max 0.1。
- 计算流程概述：h[0]←initial_state（或 0）→ 每 chunk 四段：C1 `v_work=w@h_i`（Cube mmad fp32）→ V1 `v_new=u−v_work`、乘 decay、双写出（v_new 原样 + v_up 衰减 cast 后供 C2）→ C2 `h_work=kg^T@v_up`（Cube mmad fp32）→ V2 `h_{i+1}=h_i·e^{g_last}·2^{gk_last}+h_work`。
- 锚点：`op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp:508-1067`（Process 主循环）、`op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_{vnew,update}.hpp`。

## 2. 原型与约束（详设 §2 / §2.1）

### 输入输出（来源：op_host/chunk_gated_delta_rule_fwd_h_def.cpp:42-94；aclnn CheckShape aclnn_chunk_gated_delta_rule_fwd_h.cpp:112-169）

| 参数名 | 入参类型 | 必选性 | dtype | format | shape | 语义 |
|---|---|---|---|---|---|---|
| k | 输入 | 必选 | fp16/bf16 | ND | [B,H,T,K] | key（useGk 时已被调用方预衰减为 kg） |
| w | 输入 | 必选 | 同 k | ND | [B,HV,T,K] | delta rule 的 w（逐 v-head） |
| u | 输入 | 必选 | 同 k | ND | [B,HV,T,V] | value |
| g | 输入 | 可选 | fp32 或同 k | ND | [B,HV,T] | 标量门，自然对数域，chunk 内累积 |
| gk | 输入 | 可选 | fp32 或同 k | ND | [B,HV,T,K] | 逐 K 门，log2 域，chunk 内累积 |
| initial_state | 输入 | 可选 | fp16/bf16/fp32 | ND | [N,HV,K,V] | 起始状态 |
| cu_seqlens | 输入 | 可选 | int64 | ND | [N+1] | 变长；非空则 B=1、token 展平 |
| chunk_indices | 输入 | 可选 | int64 | ND | — | 透传未用 |
| h | 输出 | 必选 | 同 k | ND | [B,HV,Nc,K,V] | 每 chunk 起始状态 |
| v_new | 输出 | 必选 | 同 k | ND | [B,HV,T,V] | 修正后 value（**未衰减**） |
| final_state | 输出 | 可选 | 同 initial_state（无则 fp32） | ND | [N,HV,K,V] | 最终状态 |

### 属性（def.cpp:96-103）

| 属性 | 类型 | 默认 | 语义 |
|---|---|---|---|
| output_final_state | bool | false | 是否公开输出 final_state |
| chunk_size | int | 64 | chunk 长（README 声明支持 64/128） |
| logical_batch/seqlen/k_heads/v_heads/k_dim/v_dim | int ×6 | — | l0op 从 shape 现场推导（op_api/chunk_gated_delta_rule_fwd_h.cpp:60-67） |

### 原型约束

- **g/gk 至少一个**（tiling.cpp:91-93；aclnn :229-234）；同给时 dtype 一致（:183-186）。
- `HV ≥ H 且 HV % H == 0`（GQA 头分组，`kHeadIdx = vHeadIdx / (HV/H)`，scheduler :213,267）。
- `vHeadDim ≤ 256`（tiling.cpp:140-144）；k/w/u 同 dtype 且 fp16/bf16；gate fp32 或同 k；state 任意 fp16/bf16/fp32。
- 变长时 shapeBatch=1、tokenBatch=len(cu_seqlens)−1，空序列跳过（scheduler InitRuntime :185-203）。
- 全部输入 AutoContiguous（def.cpp:45-82）。

## 3. 规格约束（详设 §3）

| 类别 | 取值 | 来源锚点 |
|---|---|---|
| 支持 NPU | ascend910b / 910_93 / 950（A2/A3/A5） | def.cpp:116-118；README.md:87 |
| kernel 类型 | MIX_AIC_1_2（1 Cube : 2 Vector） | op_kernel/chunk_gated_delta_rule_fwd_h.cpp:119,123 |
| dtype 全集 | 数据 fp16/bf16；gate fp16/bf16/fp32；state fp16/bf16/fp32 | def.cpp:23-37 |
| format | 全 ND，AutoContiguous | def.cpp:38-40 |
| 动态 shape | DynamicRank/DynamicShape/DynamicCompileStatic 全 true | def.cpp:106-114 |
| 交付形式 | aclnn 两段式 + PTA `torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h` | tests/pta/test_fwd_h.py:323 |

### 架构差异（A2 vs arch35，两目录并存）

| 点 | arch35（主线） | A2（op_kernel/ 顶层） |
|---|---|---|
| ArchTag | Ascend950 | AtlasA2（epilogue policies.hpp:18 vs 22） |
| scheduler | taskNum>核数时 2 流 ping-pong | 固定 1 流（短尾 chunk 非对称退役无反向信用，gemm/block/block_scheduler_gdn_fwd_h.hpp:214-216 注释） |
| update epilogue | fp32 state 直通（useFp32StateUpdate，中间 chunk 也保 fp32） | 无直通 |
| UB 偏移 | g 系在 168K+ | 160K+（整体低 8K） |

## 4. 分核策略（详设 §4.1.1 → Triton grid 设计）

- Ascend C 事实：`blockDim = aicCoreNum`（tiling_processor.h:91）；任务 = (batch, vHead) 对，`taskNum = batch·vNumHead`，chunk 维**流内串行**（scheduler :212-222）；每核 2 流 ping-pong 摊销 CV 握手空泡。
- **Triton 决策**：
  - 任务轴选择：**(b, hv, v_block) 三元组**。关键结构观察——**V 维列条带完全独立**：h[:, v_block] 的递推只依赖 u/v_new 的同列条带（C1 按 n=vBlockDim 切、C2 同理，kernel .hpp:556,682），因此比源码的 (b,h) 任务多一级并行，且 h 条带可整轮常驻片内。
  - 任务块总量公式：`NUM_PROGRAMS = N_seq · HV · (V / BLOCK_V)`（dense 时 N_seq=B）。
  - 核分配范式：**一块一 program 即可**（每 program 是长串行递推，任务数 ≈ B·HV·NVB ≫ 核数时自然占满；不属于 elementwise 家族的超额 block 负引导场景——那个教训针对的是小块海量任务）。chunk 循环在 program 内 `for i in range(nt)`。
  - 变长：grid 按 N_seq=紧凑序列数；bos/eos/chunk_off 由 wrapper 预计算（cumsum），kernel 标量 load——**替代源码的逐流 O(tokenBatch) 重扫**（scheduler ResolveVarlenSequence :229-254）。

## 5. BLOCK / Tiling 参数（详设 §4.1.2 / §4.1.3）

### Ascend C tilingdata → Triton 超参映射

| tilingdata 字段 | 语义 | host 计算式 | Triton 对应 | 候选值/推导式 |
|---|---|---|---|---|
| chunkSize | chunk 长 | attr | `BT: tl.constexpr` | {64, 128} |
| kHeadDim | K 维 | shape | `K: tl.constexpr`（dot 的 K/M 维） | next_pow2(K)，主线 128 |
| vHeadDim | V 维 | shape | `BLOCK_V: tl.constexpr` + NVB | min(next_pow2(V), 128)；V=256 → NVB=2 |
| blockTokens（scheduler :325） | 末 chunk 尾长 | batchTokens − i·chunk | 运行时 `actual = min(BT, remain)`，全程 mask | tl.dot 按 BT padding，pad 行贡献恒 0 |
| useG/useGk | gate 组合 | tensor 非空 | `USE_G/USE_GK: tl.constexpr` | 4 组合（00 由 wrapper 拦截） |
| isVariedLen/shapeBatch/tokenBatch | 变长三元 | tiling_processor.h:79-89 | `VARLEN: tl.constexpr` + cu/co 张量 | — |
| workspace 六段 offset | CV 中介 | tiling_processor.h:97-121 | **不需要**（h 条带片内常驻，v_up 片内直供 dot） | — |

### tilingkey / 模板分支 → constexpr

| tilingkey | 分支语义 | Triton 处理 |
|---|---|---|
| 1（V≤128，L1 tile 128×128×128） | TileShapes128 | 不进 kernel：BLOCK_V/NVB 由 wrapper 定 |
| 2（V=256，L1 128×256×128、L0 128×256×64） | TileShapes256 | 同上（NVB=2） |
| dtype 三级分派（state→gate→gk/g） | 模板实例 | 指针 dtype 自然携带 + USE_G/USE_GK constexpr |

## 6. 片上内存预算（详设 §5）

| Ascend C buffer | 位置 | 容量表达式 | Triton 等效 tile | 字节校核（K=128, BT=128, BV=128, fp32 h） |
|---|---|---|---|---|
| L1 tile（mmad） | L1 | 128×128×128 / 128×256×128 | tl.dot 内部管理 | 不占 UB |
| ubVWork/HUpdate ping+pong | UB 32K/96K | chunk×V×4B×2 | **不存在**（无 workspace） | 0 |
| epilogue calc/ping/pong 系 | UB 0-176K | 见 extraction §5 表 | h 条带 fp32 [K,BV] 常驻 | 64KB |
| — | — | — | w/k tile [BT,K] fp16 ×2 | 64KB（瞬态） |
| — | — | — | u/v_new/v_up [BT,BV] ×2~3 | 48-96KB（瞬态） |

- UB 预算合计：峰值 ~160-190KB vs 上限 192KB/核——**BT=128 且 K=128 时贴顶，建议 BT=128 场景降 BLOCK_V=64 或 dot 分两段 K**；BT=64 时 ~120KB 安全。
- double buffer：AscendC 全链路 ping-pong（UB 偏移表见 extraction §5）；Triton **不要 num_stages**（Triton-Ascend 运行时不生效，runbooks/autotuning/triton-ascend-microkernel-autotune-noise.md），掩盖交给编译器。

## 7. 流水编排与 Triton 可达性（详设 §6）

### 流水结构

`SyncAll` → AIV 初始化全部 h[0] → `SyncAll` → 每 chunk 四段 C1→V1→C2→V2 经 4 对 CrossCoreFlag 握手滚动（cube1Done/vec1Done/cube2Done/vec2Done，scheduler :125-128），每核 2 条 (b,h) 流 ping-pong 填空泡；direct-UB 模式（K=V=128、非变长、chunk≤64、任务≥核数）下 C1/C2 直写 UB 省 GM 往返（kernel :248-252）。

### 可达性判定表

| Ascend C 手段 | 锚点 | 可达性 | Triton 写法 / 降级 | 预期影响 |
|---|---|---|---|---|
| Catlass Mmad pingpong（L1/L0 tile） | kernel .hpp:113-133（类型别名；本体出仓） | ❌（黑盒） | `tl.dot` | Cube 内流水由编译器接管 |
| C1→V1→C2→V2 跨核四段流水 + CrossCoreFlag | kernel .hpp:531-801,939-1028 | ❌（机制）/✅（语义） | **单 program 顺序执行**（数据依赖天然满足）；并行度由 (b,hv,vb) 条带数补足 | Cube/Vector 显式交叠丢失；条带常驻省掉全部 workspace GM 往返（结构性更优） |
| Reg VF 双发射（ApplyRowScale/ComputeVNew DualIssue） | regbase.hpp:75-212 | ❌ | 普通 `*`/`-`/exp 向量算子 | 双发射由编译器调度 |
| direct-UB 直通 | kernel .hpp:248-252 | ❌ | 不需要（比它更彻底） | — |
| blockTokens<16 的 AIV 纯 FMA 兜底（ComputeTail） | kernel .hpp:353-506 | ❌（不需要） | mask+padding（pad 行贡献恒 0） | 尾 chunk 由 mmad 降级为向量——Triton 无此分叉 |
| zN 布局 v_update 供 L1 | kernel .hpp:110 | ❌ | 不需要 | — |
| AIV 开场 h[0] 初始化 + SyncAll | kernel .hpp:814-900 | ✅ | 并入主 kernel：i=0 先 store h | — |
| h_i 的 GM 读（C1 的 B 操作数） | kernel .hpp:554-558 | ⚠️ 性能风险 | h 条带片内常驻，**零 GM 读** | 每 chunk 省 K×V×2B×2 的 GM 流量 |
| k 的 ColumnMajor 读取（C2 的 A 操作数 [K,BT]） | kernel .hpp:104,124 | ⚠️ 性能风险 | **转置寻址 load [K,BT]**（`kk[:,None]+tt[None,:]*K`）；**禁用 tl.trans**（标量化 ~15x，runbooks/compute-pipeline/triton-ascend-trans-scalarization.md） | 列主序离散访问，实测定；劣化则考虑 k 预转置 workspace |
| exp2 语义 = ×ln2+Exp | regbase.hpp:66-69 | ✅ | `tl.exp(gk_last * 0.6931471805599453)` | 与源码逐位同构；`tl.exp2` 是否更优**未实测**（两态，勿断言） |

## 8. Triton 骨架（生成用伪码）

```python
# grid = (N_seq * HV * NVB,)；每 program 一条 (b, hv, v_block) 状态条带
# constexpr: K, V, BT(chunk), BLOCK_V, USE_G, USE_GK, USE_INITIAL, STORE_FINAL, VARLEN
@triton.jit
def _fwd_h(k_ptr, w_ptr, u_ptr, g_ptr, gk_ptr, is_ptr, h_ptr, v_new_ptr, fs_ptr,
           cu_ptr, co_ptr, HV, G, T, <各张量 strides...>,
           K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, ...,
           BLOCK_V: tl.constexpr):
    pid = tl.program_id(0)
    NVB: tl.constexpr = V // BLOCK_V
    vb = pid - (pid // NVB) * NVB          # 禁 %（后端契约）
    hv = (tmp := pid // NVB) - (tmp // HV) * HV
    b = tmp // HV
    hk = hv // G                           # GQA 头分组（scheduler:267）
    # VARLEN: bos/t_total/chunk_off/nt 从 cu/co 标量 load；输入寻址用 sb(=0)，state 用 b
    kk = tl.arange(0, K); tt = tl.arange(0, BT); vv = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    h = tl.load(is_ptr + b*sib + hv*sih + kk[:,None]*V + vv[None,:]).to(tl.float32) \
        if USE_INITIAL else tl.zeros((K, BLOCK_V), tl.float32)
    for i in range(0, nt):
        t0 = bos + i * BT
        actual = tl.minimum(BT, t_total - i * BT)      # 尾 chunk 真实行数
        tmask = tt < actual                            # int32 mask（后端契约）
        # 存 chunk i 起始状态（RTNE 回 input dtype）
        tl.store(h_ptr + sb*shb + hv*shh + (chunk_off+i)*shc + kk[:,None]*shk + vv[None,:],
                 h.to(out_dtype))
        # C1+V1：v_new = u - w @ h（dot 输入 cast 到 input dtype，对齐 mmad fp16-in/fp32-acc）
        w_t = tl.load(w_ptr + sb*swb + hv*swh + (t0+tt)[:,None]*swt + kk[None,:],
                      mask=tmask[:,None], other=0.0)          # [BT,K]
        u_t = tl.load(u_ptr + sb*sub + hv*suh + (t0+tt)[:,None]*sut + vv[None,:],
                      mask=tmask[:,None], other=0.0)          # [BT,BV]
        v_new = u_t.to(tl.float32) - tl.dot(w_t, h.to(out_dtype))
        tl.store(v_new_ptr + sb*svb + hv*svh + (t0+tt)[:,None]*svt + vv[None,:],
                 v_new.to(out_dtype), mask=tmask[:,None])
        # V1 衰减：g_last 必须取 actual-1 行（⚠️ 正确性风险，取错静默错值）
        if USE_G:
            g_t = tl.load(g_ptr + sb*sgb + hv*sgh + t0 + tt, mask=tmask, other=0.0).to(tl.float32)
            g_last = tl.load(g_ptr + sb*sgb + hv*sgh + t0 + actual - 1).to(tl.float32)
            v_up = v_new * tl.exp(g_last - g_t)[:, None]
            h = h * tl.exp(g_last)
        else:
            v_up = v_new
        if USE_GK:
            gk_last = tl.load(gk_ptr + sb*sgkb + hv*sgkh + (t0+actual-1)*sgkt + kk).to(tl.float32)
            h = h * tl.exp(gk_last * LN2)[:, None]     # exp2 语义 = ×ln2+exp（regbase.hpp:66-69）
            # kg = k（KDA 逐 key 预衰减由调用方完成）
        # C2：h += k^T @ v_up；k 转置寻址 load，禁 tl.trans
        k_t = tl.load(k_ptr + sb*skb + hk*skh + (t0+tt)[None,:]*skt + kk[:,None],
                      mask=tmask[None,:], other=0.0)          # [K,BT]
        h += tl.dot(k_t, v_up.to(out_dtype))
    if STORE_FINAL:
        tl.store(fs_ptr + b*sfb + hv*sfh + kk[:,None]*V + vv[None,:],
                 h.to(fs_ptr.dtype.element_ty))               # fp32 state 直通天然成立
```

wrapper 要点：① g/gk 至少一个（对齐 tiling.cpp:91-93）；② NVB=V//BLOCK_V，BLOCK_V=min(128, next_pow2(V))；③ varlen 预计算 cu/co（cumsum）；④ 输出分配 h[B,HV,Nc,K,V]（varlen B=1、Nc=总 chunk 前缀和）、v_new、final_state（dtype=initial_state 或 fp32）；⑤ `state_v_first` 用 `transpose(-1,-2)` 零拷贝（勿复刻 aclnn TransposeLastTwo）；⑥ 全 shapes/strides 传参，无 autotune、无 num_stages；⑦ contiguous 断言。

完整可运行版为转化工作流的过程产物 `triton_skeleton.py`，结构与本节一致（含 varlen 与 golden 复刻 `ref_chunk_gated_delta_rule_fwd_h`）。

## 9. 精度标杆与验收（详设 §测试设计）

- 标杆类别：单标杆——`forward_h_trans_cpu`（tests/pta/test_fwd_h.py:39-138）全 fp32 torch 递推。
- 标杆实现要点：chunk 内 cumsum 的 g；`S[i+1] = S[i]·exp(g_last) + k^T@(v_new·exp(g_last−g))`；末 chunk 写 final_state；输出 cast 回 input dtype。
- **算子级验收标准**（对齐官方比较器 data_compare_h.py:133,195-205）：h / v_new / final_state 三输出分别判定——**相对误差阈值 1e-4、≥99.99% 元素达标、单点 max_diff < 0.1**；g-only 路径为官方 golden 覆盖域，gk/both 路径为扩展参考（golden 未覆盖，验收时建议对 KDA 公式自建 fp32 标杆）。
- **benchmark 语义超集检查**：PTA 测试支持 actual input/output 对照（:194-219）、变长随机切分（:221-230）；无 CANN ST json（非 CANN 仓）。
- 典型用例 shape：HQ=HK=16、HV=32（:52 注释）；K=V=128、chunk=64 为主线；V=256/chunk=128 为扩展。
- 边界用例：尾 chunk（T%chunk≠0，blockTokens<16 触发源码向量兜底路径）；空序列（变长 len=0 跳过）；无 initial_state（h[0]=0）；state fp32 + input fp16 的混合精度。
- **填充纪律**：g 必须 chunk 内 cumsum 的单调负衰减（base∈[−15,−5] + U(−2,0)，gen_decay_data:246-262）；无结构随机 g 会让 exp(g_last−g_t) 指数爆炸，超出 1e-4 容差判定域。

## 10. Agent actions（生成指引）

1. 识别算子类别：CV 融合 + 状态递推（recurrent）+ 双 GEMM；kernel 类型 MIX_AIC_1_2。
2. 切分轴取 **(b, hv, v_block) 状态条带**（V 列条带独立是关键结构观察），grid=N·HV·NVB；chunk 维 program 内串行——勿把 chunk 摊到 grid（h 依赖串行）。
3. BLOCK：BT=chunk_size、K=next_pow2(K)、BLOCK_V=min(128, next_pow2(V))；BT=128&K=128 时 BLOCK_V 降 64 保 UB（§6）。
4. 精度提升点：h 条带 fp32 常驻；v_new/decay fp32 域；dot 输入 cast 到 input dtype（对齐 mmad 语义）；store 单次 RTNE；**g_last/gk_last 取 actual−1 行**。
5. 不可达点降级：Catlass mmad→tl.dot；CrossCoreFlag 流水→单 program 顺序；Reg VF→普通向量算子；**禁 tl.trans**（k 用转置寻址 load）。
6. **勿复刻源码侧分解**：GM workspace 六段（vWork/vUpdate/kDecay/hWork/numSeq/numChunks）全部不需要；CrossCoreFlag/EVENT_ID 事件组不需要；ComputeTail 向量兜底不需要（mask 取代）；aclnn TransposeLastTwo 用 stride 交换取代；**两态保留**：h 条带常驻要求 K×BLOCK_V×4B ≤ ~64KB，K>128 场景需改 h 分段外循环（⚠️ 未实测，超 K=128 时先验证）。
7. 无 autotune、无 num_stages（Triton-Ascend 不生效）；BLOCK 全部按 §5 静态推导。
8. 验收按 §9 标准执行（三输出、rel 1e-4 / 99.99% / max 0.1）；g 填充遵守 cumsum 纪律。

## 11. 复现性声明

本卡由 Ascend C 源码静态分析转化（confidence: inferred）。未经硬件实测的点：① 转置寻址 load [K,BT] 的性能（vs 预转置 workspace 两态）；② K>128 时 h 条带常驻的 UB 可行性；③ tl.exp2 vs ×ln2+exp 差异；④ KB 组形态（gk/both）golden 未覆盖，公式以 README+epilogue 源码为准。
回放验证：未做（P4 待 benchmark case 与生成环境）。
**依赖缺口**：Catlass Mmad block 与 kernel_utils regbase 本体不在快照（manifest 缺省项 4），mmad 内部流水按黑盒；如需补齐从 fla_npu 仓 third_party/ 或 Catlass 上游获取。

**消费反馈（已知实例）**：Gdn2Fwd（varlen Gated DeltaNet 2 前向，ascend910b1，2026-09-14）设计期 strong 命中本卡——chunk 间状态递推的 (b,hv,v_block) 条带常驻并行与状态写回纪律直接被其双骨架中的 chunked WY 路径复用；实测案例见 [Gdn2Fwd 双骨架 host dispatch 实测案例](gdn2-fwd-chunked-wy-dispatch.md)。

<!-- okf:related:start -->

# 相关

- 实践案例: [MatMul / GEMM 高性能算子](../reference/op-class-matmul.md) — 主题相关：共享 matmul, fp32_accum——FwdH 的 kg^T@(v_new·decay) 等矩阵乘核心与 GEMM 类同源
- 实践案例: [SSM / 状态空间模型算子优化](../reference/op-class-ssm.md) — 同属 SSM/线性注意力递推算子类：FwdH 即 chunk 间状态递推
- 实践案例: [GEMM 操作数保持 low_dtype，fp32 累加](../runbooks/algorithm-numerics/gemm-low-dtype-fp32-accum.md) — FwdH 全 fp32 累加实践与该 low_dtype/fp32 累加契约一致
- 实践案例: [chunk 递推算子的依赖二分法与 Pingpong 下发调度](../runbooks/compute-pipeline/chunk-recurrence-pingpong-dispatch.md) — 主题相关：共享 gdn
- 实践案例: [matmul core mapping 策略选择（列/行/对角）](../runbooks/compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 实践案例: [依赖无关输入预加载（load 提前于等待边）与带宽仲裁意识](../runbooks/compute-pipeline/cv-preload-and-bandwidth-arbitration.md) — 主题相关：共享 gdn
- 实践案例: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../runbooks/compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 实践案例: [Scheduling Optimization: hardware unit isolation](../runbooks/compute-pipeline/exp_fused_norm_rope_matmul_001.md) — 主题相关：共享 matmul
- 实践案例: [计算优化: fixpipe 随路量化 + K 维循环展开 + 预取](../runbooks/compute-pipeline/exp_matmul_002.md) — 主题相关：共享 matmul
- 实践案例: [负载均衡分核：grid 对齐物理核数，核内循环替代 GPU 多维 grid](../runbooks/compute-pipeline/load-balance-grid-sizing.md) — GDN 三算子分核实测出处（fwd_o/kkt +99%）
- 实践案例: [标量循环改写为 tl.dot，提升 Cube 利用率](../runbooks/compute-pipeline/scalar-loop-to-dot.md) — kda/ssm 标量循环改 tl.dot，对应 FwdH 的 kg^T 矩阵乘表达
- 实践案例: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../runbooks/memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 实践案例: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../runbooks/memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 实践案例: [访存优化：大尺寸 matmul 的 256 分块与 L0C 复用](../runbooks/memory-sram/exp_matmul_001.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：A、B 均不转置 matmul 的 512B 行宽对齐分块](../runbooks/memory-sram/exp_matmul_003.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：A 不转置、B 转置 matmul 的 K0=256 统一分块](../runbooks/memory-sram/exp_matmul_004.md) — 主题相关：共享 matmul
- 实践案例: [切分优化：双转置矩阵乘的 512B 对齐分块](../runbooks/memory-sram/exp_matmul_005.md) — 主题相关：共享 matmul
- 实践案例: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](../runbooks/memory-sram/layout-bthd-to-bhtd.md) — chunkwise 递推类 layout 预转置直接适用 FwdH
- 实践案例: [Cube-Vector 场景行宽必须 512B 对齐](../runbooks/memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 实践案例: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../runbooks/memory-sram/ub-fill-and-compute-hoist.md) — GDN dqkwg UB 打满实践，chunk_size_128 同参数域
- 实践案例: [tl.dot vs torch.matmul 精度差异诊断](../runbooks/meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 实践案例: [流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段](../runbooks/meta-skills/pipeline-diagram-driven-diagnosis.md) — Fwd_h 为流水图气泡分类方法论的目标算子
- 实践案例: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡](add-broadcast-alpha.md) — 主题相关：共享 bf16, fp16
- 相关主题: [ApplyRotaryPosEmb（RoPE 旋转位置编码 q/k 双路融合）Ascend C → Triton 语义转化卡](apply-rotary-pos-emb.md) — 主题相关：共享 bf16, fp16
- 相关主题: [Elementwise absolute value (Abs)](elementwise-abs.md) — 主题相关：共享 bf16
- 相关主题: [Elementwise add with a tail mask](elementwise-add.md) — 主题相关：共享 bf16
- 相关主题: [FlashAttentionScoreGrad training-backward on Ascend NPU](flash-attention-score-grad.md) — 主题相关：共享 matmul
- 相关主题: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](fused-bias-leaky-relu.md) — 主题相关：共享 fp16
- 相关主题: [GDN 线性注意力三阶段优化案例（FLA 移植 → Triton 体系化 → AscendC 重写）](gdn-chunked-linear-attention.md) — 主题相关：共享 cv_fusion, gdn, linear_attention
- 相关主题: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](gdn2-fwd-chunked-wy-dispatch.md) — 同 GDN chunk 递推前向算子族：Gdn2Fwd 为其 varlen GDN2 变体实测案例
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](moe-finalize-routing.md) — 主题相关：共享 bf16, fp16
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](moe-gating-top-k-softmax.md) — 主题相关：共享 bf16, fp16
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](moe-init-routing-quant-v2.md) — 主题相关：共享 bf16, fp16
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](moe-init-routing-quant.md) — 主题相关：共享 bf16, fp16
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](moe-token-permute-with-ep.md) — 主题相关：共享 bf16, fp16
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](moe-token-permute.md) — 主题相关：共享 bf16, fp16
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](moe-token-unpermute-grad.md) — 主题相关：共享 bf16, fp16
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](moe-token-unpermute.md) — 主题相关：共享 bf16, fp16
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](nsa-compress-with-cache.md) — 主题相关：共享 bf16, fp16
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](recurrent-gated-delta-rule.md) — 同 gated delta rule 算子族：RGDR 逐步递推 vs FwdH chunkwise 并行形式
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](ring-attention-update.md) — 主题相关：共享 bf16, fp16
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](rmsnorm-rope.md) — 主题相关：共享 bf16

<!-- okf:related:end -->
