---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）Ascend C → Triton 语义转化卡
description: GDN/KDA 共享的 chunk 间状态递推算子：v_new = u − w@h_prev，h_next = h_prev·exp(g_last)·exp2(gk_last) + kg^T@(v_new·decay)。CV 融合（MIX_AIC_1_2，Catlass 模板）+ 双流 ping-pong + 全 fp32 中间域。对 Triton 生成的关键决策：(b,hv,v_block) 状态条带常驻并行轴、V 维列条带独立性消解全部 GM workspace、persistent-core grid=num_aicore、pow2/mask 泛化覆盖非 2 幂 K/V、禁 tl.trans 用转置寻址 load、g 的 chunk 内 cumsum 语义与 g_last 取 actual-1 行。
tags: [gdn, kda, gated_delta_rule, linear_attention, recurrent_state, cv_fusion, matmul, fp16, bf16, fp32_accum, varlen, chunk_gated_delta_rule_fwd_h]
confidence: inferred
status: draft
resource: github.com/jiangdm2024/flash-linear-attention-npu_dev @ 4c78653f47beda77681aa725c7fa49c7ce8d86a9（fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h；本地快照 zmm_source_retrieval/src/chunk_gated_delta_rule_fwd_h/）
created_at: "2026-09-09T00:00:00Z"
updated_at: "2026-09-14T00:00:00Z"
---

sources:
  - url: github.com/jiangdm2024/flash-linear-attention-npu_dev @ 4c78653f47beda77681aa725c7fa49c7ce8d86a9
    role: primary

# ChunkGatedDeltaRuleFwdH（GDN/KDA chunk 间状态递推前向）

> 材料：`output/chunk_gated_delta_rule_fwd_h/{manifest,extraction,mapping}.md`（2026-09-14
> 新流程复验版；2026-09-09 旧流程初版在 `baseline_20260909/`）。
> 本卡外部论断只挂引用不复写（维护纪律 README:83）：后端契约 → `backend_facts/backend_contracts.md`；
> 平台常数 → `backend_facts/hardware_platforms.md`；dtype 陷阱 → `backend_facts/dtype_support.md`；
> 编译选项 → `backend_facts/compiler_options.yaml`；算子范式 → `backend_facts/operator_class_patterns.md`。

逐 chunk 递推线性注意力的隐状态：`v_new = u − w@h_prev`；
`h_next = h_prev·scalar_decay·key_decay + kg^T@(v_new·decay)`，输出每个 chunk 的起始状态 h
（反向要用）、修正 v_new、可选 final_state。kernel 类型 **CV 融合**
（`KERNEL_TASK_TYPE(*, KERNEL_TYPE_MIX_AIC_1_2)`，chunk_gated_delta_rule_fwd_h.cpp:119,123，
2026-09-14 复验），基于 Catlass 模板库。**依赖缺口**：Catlass Mmad block 本体不在快照
（manifest 缺省项 4，仍成立），mmad 内部流水按黑盒；GDN 专用 regbase epilogue VF
**在快照内**（block_epilogue_gdn_fwdh_regbase.hpp，exp2 语义证据链完整——旧版表述过宽已修正）。

## 1. 功能与公式

- 功能：GDN/KDA 前向的 chunk 间状态递推（h propagation），供后续 chunk 内注意力使用。
- 计算公式（每 batch n、v-head hv、chunk i；README.md:20-27 + golden test_fwd_h.py:111-115 ✅复验）：

$$\text{v\_new}[t] = u[t] - w[t]\, h_i, \qquad \text{decay}[t] = e^{\,g[\text{last}]-g[t]}$$
$$h_{i+1} = h_i \cdot e^{\,g[\text{last}]} \cdot 2^{\,g_k[\text{last},:]} \;+\; k_g^{\top}\big(\text{v\_new}\odot \text{decay}[:,None]\big)$$

  变量：$h_i \in \mathbb{R}^{K\times V}$；$g$ 标量门（**自然对数域，chunk 内 cumsum 语义**）；
  $g_k$ 逐 K 门（**log2 域**）；$g[\text{last}]$ = chunk 内最后有效 token（**actual−1 行**）；
  $k_g$ = useGk 时调用方预衰减的 k（vnew epilogue:407 ✅复验），否则 = k。末 chunk 的
  $h_{i+1}$ 写入 final_state。
- **必须保留的语义点**：
  1. g 是 chunk 内**累积**量（cumsum，gen_decay_data:261）——g[last] 是整 chunk 总衰减
  2. g_last/gk_last 取 **actual−1 行**（尾 chunk 不是 BT−1）——⚠️ 取错静默错值
  3. **exp2 语义 = ×LN2+Exp**（regbase.hpp:53-73 ✅复验）；useExp2 模板恒 false
  4. h[i] 存 chunk i 的**起始**状态（chunk0 = initial_state 或 0）
  5. kg 的预衰减**只在调用方发生一次**，kernel 内 k 不再乘 decay
  6. 中间域全 fp32；dot 输入为 input dtype（mmad fp16-in/fp32-acc）；写出单次 CAST_RINT
- **语义分层**：GEIR kernel 本体 = 上式；aclnn 接口层消化 state_v_first 的 [K,V]↔[V,K]
  （TransposeLastTwo，Triton 用 stride 交换零拷贝替代）与 output_final_state=false 的不可见
  finalStateCompute（Triton h 条带寄存器常驻，**不需要**）。

## 2. 原型与约束

### 输入输出（def.cpp:42-94；aclnn CheckShape :112-169）

| 参数名 | 必选性 | dtype | shape | 语义 |
|---|---|---|---|---|
| k | 必选 | fp16/bf16 | [B,H,T,K] | key（useGk 时已被调用方预衰减为 kg） |
| w | 必选 | 同 k | [B,HV,T,K] | delta rule 的 w |
| u | 必选 | 同 k | [B,HV,T,V] | value |
| g | 可选 | fp32 或同 k | [B,HV,T] | 标量门，自然对数域，chunk 内累积 |
| gk | 可选 | fp32 或同 k | [B,HV,T,K] | 逐 K 门，log2 域，chunk 内累积 |
| initial_state | 可选 | fp16/bf16/fp32 | [N,HV,K,V] | 起始状态 |
| cu_seqlens | 可选 | int64 | [N+1] | 变长；非空则 B=1、token 展平 |
| h | 必选输出 | 同 k | [B,HV,Nc,K,V] | 每 chunk 起始状态 |
| v_new | 必选输出 | 同 k | [B,HV,T,V] | 修正后 value（**未衰减**） |
| final_state | 可选输出 | 同 initial_state（无则 fp32） | [N,HV,K,V] | 最终状态 |

### 原型约束

- **g/gk 至少一个**（tiling.cpp:91-93）；同给时 dtype 一致
- `HV ≥ H 且 HV % H == 0`（GQA，`kHeadIdx = vHeadIdx / (HV/H)`，scheduler :267 ✅复验）
- `vHeadDim ≤ 256`（tiling.cpp:140-144）；README 声明 V∈{128,256}、chunk∈{64,128}——
  ⚠️ **官方 benchmark 是其超集**：85_/kernelbench 50 case 的 K∈{7..80}、V∈{5..50}、
  chunk∈{16,32,64}（非 2 幂！），Triton 实现**必须 pow2/mask 泛化**（§5）
- 变长时 shapeBatch=1，空序列跳过（scheduler InitRuntime :185-203）
- 全部输入 AutoContiguous

## 3. 规格约束

| 类别 | 取值 |
|---|---|
| 支持 NPU | ascend910b / 910_93 / 950 |
| kernel 类型 | MIX_AIC_1_2（1 Cube : 2 Vector，.cpp:119,123 ✅复验） |
| dtype | 数据 fp16/bf16；gate fp16/bf16/fp32；state fp16/bf16/fp32 |
| tilingkey | 1（V≤128）/ 2（V=256）→ Triton 不进 kernel（BV_TILE/NVB wrapper 定） |
| workspace | 源码六段（vWork/vUpdate/kDecay/hWork/numSeq/numChunks）→ **Triton = 0** |
| 交付 | aclnn 两段式 + PTA `torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h` |

### 架构差异（A2 vs arch35）

arch35：taskNum>核数时双流 ping-pong、fp32 state 直通（useFp32StateUpdate）、UB g 系偏移
168K+；A2：固定单流（短尾 chunk 非对称退役无反向信用）、无 fp32 直通、偏移低 8K。
**Triton 无此分叉**（无跨核流概念；fp32 直通由条带常驻天然成立）。

## 4. 分核策略（→ Triton grid）

- Ascend C 事实：`blockDim = aicCoreNum`；任务 = (batch, vHead) 对，chunk 维流内串行；
  每核 2 流 ping-pong 摊销 CV 握手空泡。
- **Triton 决策**：
  - 任务轴：**(b, hv, v_block) 条带**——V 维列条带完全独立（C1/C2 按 n=vBlockDim 切，
    kernel .hpp:556,682），比源码多一级并行；h 条带 [BK, BV_TILE] fp32 常驻片内，
    整轮递推不出 program。
  - **grid = min(NUM_TASKS, num_aicore)**，persistent-core
    `for task in range(pid, NUM_TASKS, NUM_CORE)`（**C-011**：含 tl.dot = CV 融合取 Cube
    数 [DOC] `operator_class_patterns.md`）。核数运行时查询、**禁止写死**
    （`hardware_platforms.md` §3：950 AIC 4~36 跨度 9 倍）。
  - ⚠️ 两态备注：旧版「一块一 program」（grid=NUM_TASKS）+ MCTS 910b 实测 7.23x
    （src/chunk_gated_delta_fwd_h-mcts-mcts-0901）是 P4 可对照的另一态；默认按 C-011
    persistent-core（对长串行任务零成本，且免除海量 program launch 风险）。
  - 变长：cu/co 由 wrapper 预计算（torch.cumsum），kernel 标量 load（O(1) 替代源码
    逐流 O(tokenBatch) 重扫，scheduler :229-254）。

## 5. BLOCK / Tiling 参数

| tilingdata | Triton 超参 | 推导 |
|---|---|---|
| chunkSize ∈ {16,32,64,128} | `BT: tl.constexpr` | 全 chunk 单 tile（均 2 的幂） |
| kHeadDim | `BK: tl.constexpr` + kmask | **max(16, next_pow2(K))**——tl.arange 须 2 的幂、tl.dot 最小维 16；官方 50 case 的 K 非 2 幂，**mask 泛化是跑通正确性套件的前提**（mapping §0.6-3） |
| vHeadDim ≤ 256（benchmark 5~256） | `BV_TILE: tl.constexpr` + NVB + vmask | **max(16, min(BLOCK_V, next_pow2(V)))**；V=256 → NVB=2 |
| blockTokens（scheduler :325） | 运行时 `actual` + tmask | tl.dot 按 BT padding，pad 行贡献恒 0 |
| useG/useGk/initial/final/varlen | `USE_G/USE_GK/USE_INITIAL/STORE_FINAL/VARLEN: tl.constexpr` | 编译期裁剪；「都不给」wrapper 拦截 |

**BLOCK_V 平台分态**：950（UB 248KB [SRC]）BT=128/BK=128/BV=128（~190KB）可行；
910b（192KB）同配置贴顶 → **BLOCK_V=64**（与 MCTS 910b1 冠军 BV=min(64,V) [EXP] 一致）。

## 6. 片上内存预算

| tile（fp32 计） | BT=128,BK=128,BV=128 | BT=64,BK=128,BV=64 |
|---|---|---|
| h 条带常驻 | 64KB | 32KB |
| w/k tile ×2（瞬态） | 64KB | 32KB |
| u/v_new/v_up ×2~3（瞬态） | 96~128KB | 48~64KB |
| **峰值** | **~190KB** | **~112KB** |
| vs 950 = **248KB** [SRC] | ✅ 富余 1.3 倍 | ✅ |
| vs 910b = 192KB | ⚠️ 贴顶 → BV=64 | ✅ |

双缓冲由 multibuffer 承接（bishengir 默认 true [SRC] `compiler_options.yaml`；本算子
独立 [EXP] 交叉验证：MCTS opt-round-5 探针 multibuffer=False → -5.9% 后回退）。
**不配 num_stages**（C-006 运行时不生效）。

## 7. 流水编排与可达性

| Ascend C 手段 | Triton | 判定（依据） |
|---|---|---|
| C1→V1→C2→V2 跨核四段流水 + CrossCoreFlag ×4 对 | 单 program 顺序执行（数据依赖天然满足） | ❌机制/✅语义——结构性替代，GM 往返逐 chunk 可数（6+ 次 → 0 次 workspace） |
| Catlass Mmad pingpong（黑盒） | `tl.dot`（T2 Cube，`tl_op_lowering.md`） | ❌ 编译器接管 |
| Reg VF 双发射 | 普通 `*`/`-`/`tl.exp`（fp32 向量原生 [SRC] `dtype_support.md`） | ❌ 编译器调度 |
| direct-UB 直通（.hpp:248-252 ✅复验） | 不需要（比它更彻底：无 workspace GM） | ❌ |
| blockTokens<16 的 AIV FMA 兜底 / zN 布局 | mask + padding | ❌ 不需要 |
| AIV 开场 h[0] 初始化 + SyncAll | 并入主 kernel（i=0 先 store h） | ✅ |
| k 的 ColumnMajor 读（C2 A 操作数） | **转置寻址 load [BK,BT]**；**禁 tl.trans**（标量化 ~15x runbook） | ✅ ⚠️ 列主序访存性能实测定；劣化则 wrapper 预转置（未实测勿断言，C-012） |
| exp2 = ×LN2+Exp | `tl.exp(gk_last * LN2)` 逐位同构 | ✅（tl.exp2 两态未实测） |
| CV 同步 | sync_solver / auto_bind_sub_block 默认 true [SRC] | ⚠️ 编译器自动，不手写 al.sync_block_* |

## 8. Triton 骨架

完整骨架：`output/chunk_gated_delta_rule_fwd_h/triton_skeleton.py`（自包含，含 wrapper
与 golden 复刻）。结构一句话：persistent-core 外循环领 (b,hv,v_block) 条带任务 →
h 条带 fp32 初始化（initial_state 或 0）→ chunk 内循环｛store h[i] → w/u load →
dot → v_new store → g/gk 衰减 → k 转置 load → dot 累加 h｝→ 可选 final_state store。
关键纪律：BK/BV_TILE pow2+mask 泛化；int64 标量大基址 + int32 块内偏移（4e）；
g_last 取 actual−1 行；无 `%`/num_stages/autotune/tl.trans；mask 全 int32 比较。

## 9. 精度标杆与验收

- 标杆：`forward_h_trans_cpu`（test_fwd_h.py:111-115 ✅复验）全 fp32 torch 递推；
  骨架含复刻 `ref_chunk_gated_delta_rule_fwd_h`。gk/both 路径 golden 未覆盖（g-only），
  KDA 公式以 README+epilogue 为准自建标杆。
- 验收对齐官方比较器（data_compare_h.py:133,195-205）：**rel diff 1e-4、≥99.99% 元素
  达标、单点 max_diff < 0.1**，h/v_new/final_state 三输出分别判定。
- **填充纪律**：g 必须 chunk 内 cumsum 单调负衰减（base∈[−15,−5]+U(−2,0)）；
  无结构随机 g → exp(g_last−g_t) 指数爆炸出容差域。
- **benchmark 双套件**：
  - 正确性：官方 50 case（`ascendc-kernelgen-data/npu_benchmark/level4/85_ChunkGatedDeltaRuleFwdH.{py,json}`
    ≈ `npu-kernelbench/.../ChunkGatedDeltaRuleFwdH/workload.jsonl`）——小 shape、
    **K/V 非 2 幂**、fp32/fp16/bf16、gate_mode 0/1/2、initial/final/state_v_first 组合；
    另有配套输入生成器 `evaluator/custominputs/level4/chunk_gated_delta_rule_fwd_h.py`
  - 性能：MCTS 8 大 shape（`src/chunk_gated_delta_fwd_h-mcts-mcts-0901/opt-round-5/chunk_gated_delta_fwd_h_mc.json`）——
    K=V=128 bf16、T 至 32768、varlen、GQA、chunk 64/128
- 边界用例：尾 chunk（T%chunk≠0）；空序列（varlen len=0）；无 initial_state（h[0]=0）；
  state fp32 + input fp16 混合精度。

## 10. Agent actions

1. 取 `triton_skeleton.py` 为起点（勿从零写）。识别类别：**含 tl.dot 的 CV 融合 +
   状态递推**——grid 取 `num_aicore`（不是 num_vectorcore）、persistent-core（C-011）。
2. **先核 shape 泛化**：目标 case 的 K/V 非 2 的幂时，BK/BV_TILE 的 mask 路径不可回退
   为「K/V 直接进 tl.arange」（编译即失败，mapping §0.6-3）。
3. **g_last/gk_last 必须取 actual−1 行**；g 填充遵守 chunk 内 cumsum 纪律（§9）。
4. 按平台选 BLOCK_V：950 → 128；910b → 64（§5/§6）。
5. 语义六条（§1）逐条保留：cumsum g、actual−1 行、×LN2+Exp、起始状态语义、
   kg 预衰减只一次、fp32 中间域 + 单次 RTNE。
6. 生成前过 preflight（`backend_contracts.md` 8 项 + `dtype_support.md` 4a~4e）：
   禁 `%`（kernel 内）、无 num_stages/num_warps、无 autotune 残留、mask 整型比较、
   grid=min(tasks, aicore)、**禁 tl.trans**、int64 仅限标量基址、无 bf16 向量算术。
7. 不要复现：GM workspace 六段、CrossCoreFlag/EVENT_ID 事件组、ComputeTail 兜底、
   aclnn TransposeLastTwo 与不可见 finalStateCompute（§7/§8 均有替代）。
8. 验收按 §9：先跑通官方 50-case 正确性套件，再上 MCTS 8 大 shape 测性能
   （两态对照：persistent-core vs 一块一 program，mapping §0.6-2）。

## 11. 复现性声明

本卡由 Ascend C 源码静态分析转化（confidence: inferred）。2026-09-14 新流程复验：
P0/P1 锚点抽验 6/6 有效（另修正 P1 两处：UB 192KB 无锚点误述——950 实为 248KB [SRC]；
regbase epilogue VF 在快照内）；P2 六任务检索全部命中 backend_facts 首层；骨架
preflight 13 项 0 真命中。相对 2026-09-09 旧版（`baseline_20260909/`）的修正：
① UB 预算分平台（950=248KB / 910b=192KB），「降 BV=64」更正为 910b 分态而非通用结论；
② grid 改 persistent-core（旧版「一块一 program 即可」的豁免无实测，违反 C-012）；
③ 骨架 pow2/mask 泛化（旧骨架在官方 50-case 正确性套件上编译即失败）。
未经硬件实测：转置寻址 load 性能、persistent vs 一块一 program 两态、tl.exp2 vs
×LN2+Exp、K>128 的 h 条带常驻可行性。回放验证：未做（P4 待执行）。
**历史生成参照**：MCTS 5 轮实验（910b1，2026-09-01）geomean 7.2335x vs torch
（BV=64、multibuffer=True、一块一 program、8 大 shape）——其最优配置与本卡 910b
分态建议一致，可作 P4 性能刻度。**依赖缺口**：Catlass Mmad block 本体不在快照，
mmad 内部流水按黑盒。

## 维护记录

- 2026-09-09 初版（旧流程，confidence: inferred）。
- 2026-09-14 新流程重提取：外部论断改挂 backend_facts 引用；修正 mapping §0.6 三项
  （UB 分平台、grid persistent-core、pow2/mask 泛化）；骨架同步修订（另：STORE_FINAL
  两态化，去除恒分配 fs）。本算子**不是** backend_facts 契约的来源算子，且历史 MCTS
  实验（9-01）早于旧卡（9-09）与 backend_facts（9-11），三方互相独立——本次修正
  构成对 backend_facts 的独立命中验证，其 multibuffer/BV 结论与 MCTS [EXP] 互证。
