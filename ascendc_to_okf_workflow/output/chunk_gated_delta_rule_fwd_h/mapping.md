# P2 语义转化 — chunk_gated_delta_rule_fwd_h（Ascend C → Triton，新流程版）

> 输入：extraction.md（2026-09-14 复验版）。每条 P1 事实在此有去向（转化/合并/显式丢弃+理由）。
> 可达性三态：✅ 直接可达 / ⚠️ 编译器自动或需权衡 / ❌ 不可达（含降级方案）。
> **与 2026-09-09 旧版（baseline_20260909/mapping.md）的差异**：本版按『参考源与检索指引』
> 对六类后端论断强制取证（backend_facts → zh → IR 源码），每条标 [EXP]/[DOC]/[SRC]；
> §0.5 为六任务检索记录；旧版 2 处错误论断 + 1 处骨架潜伏缺陷在本版修正（见 §0.6）。
>
> **核心结论（与旧版一致，予以维持）**：任务轴取 **(b, hv, v_block) 状态条带**——V 维列
> 条带完全独立（C1/C2 都按 n=vBlockDim 切，kernel .hpp:556,682），比源码 (b,h) 任务多一级
> 并行；h 条带 [K, BLOCK_V] fp32 常驻片内，整轮递推不出 program——结构性消解 CV 四段流水、
> 全部 GM workspace、direct-UB 直通（比它更彻底）。

## 0.5 检索记录（新流程六任务，证据全标）

| # | 检索任务 | 命中 | 结论 |
|---|---|---|---|
| ① | 骨架各 tl op 的 lowering 分级 | `tl_op_lowering.md`；`dtype_support.md` 附录 A | tl.load/store = T1；**tl.dot = T2（Cube）**；tl.exp = T1/T2（fp32 VExp 向量原生 [SRC]）；标量 int32 div（任务解码 `//`）非向量 op。**禁用 tl.trans**（标量化 ~15x，runbooks/compute-pipeline/triton-ascend-trans-scalarization.md）——k 用转置寻址 load [BK,BT]。全 T1/T2，无慢路径 |
| ② | ⚠️/❌ 判定依据 | `backend_contracts.md` C-012；`dtype_support.md` §2.1 | 跨核四段流水/Catlass mmad/direct-UB 的丢弃是**结构性替代**（单 program 顺序执行天然满足数据依赖；h 条带常驻比 direct-UB 更彻底——GM 往返次数可逐 chunk 计数论证），非 C-012 等价断言；**k 的转置寻址 load [BK,BT]** 是列主序离散访存 ⚠️ 性能实测定（两态：劣化则 wrapper 预转置 k——⚠️ 未实测，勿断言） |
| ③ | 编译选项 | `compiler_options.yaml` by_operator_class.**cv_fusion**；`auto_passes.md` | 含 tl.dot → CV 融合类：enable_auto_bind_sub_block / sync_solver 默认 true [SRC Options.td]，不需手设；**multibuffer bishengir 默认 true [SRC]**——且有本算子独立 [EXP] 交叉验证：MCTS 实验（src/chunk_gated_delta_fwd_h-mcts-mcts-0901/opt-round-5）探针 multibuffer=False → 6.8053x vs True 7.2335x（-5.9%）后回退。**不写 num_stages**（C-006 [EXP]） |
| ④ | UB/对齐预算常数 | `hardware_platforms.md` | **950 UB=248KB 可用**（旧版按 192KB 校核并判「贴顶降 BLOCK_V」）；**A2/910b=192KB**。分平台重算见 §4：950 上 BT=128/K=128/BV=128（~190KB）可行；910b 上贴顶须 BV=64——与 MCTS 冠军配置 BV=64（910b1 [EXP]）吻合。fp32 计算域下 BT/BK/BV 恒 32B 倍数（C-010） |
| ⑤ | dtype 陷阱 | `dtype_support.md` §2.1/§5（含新 4e） | ❶ bf16/fp16 load 后 cast fp32 计算（950 regbase 无 bf16 向量指令 [SRC]）；dot 输入 cast 回 input dtype（对齐 mmad fp16-in/fp32-acc 语义）；❷ mask 全 int32 比较（C-002）；❸ cu_seqlens 是 int64——**仅标量 load**（`tl.load(cu_ptr+b).to(tl.int32)`），不进向量 lane [SRC i64 标量化]；❹ **4e 寻址审计**：最大 case（B=4 变长 T=32768，HV=32）h numel ≈ 1.07e9、密集 B=16 时 b·stride(0) 可达 4e9 > 2³¹ → **大基址用 int64 标量**（`b.to(tl.int64)`），块内偏移 int32（见 §6 与骨架） |
| ⑥ | 契约违禁写法 | `backend_contracts.md` C-001/C-002/C-006/C-011/C-012 | 无 `%`（解码 `vb = task - (task//NVB)*NVB`）；无 num_stages/num_warps/autotune（BLOCK 全静态推导）；**grid 修正**：旧版「一块一 program 即可」偏离 C-011 persistent-core 范式，其自我豁免（「那个教训针对小块海量任务」）是无实测等价断言、违反 C-012——本版改 **persistent-core grid = min(task_num, num_aicore)**（含 tl.dot = CV 融合取 Cube 数）；MCTS 冠军（一块一 program，910b 7.23x）记为 P4 两态对照，不作默认 |

## 0.6 对旧版的修正（2 处错误论断 + 1 处骨架潜伏缺陷）

| # | 旧版论断 | 本版修正 | 依据 |
|---|---|---|---|
| 1 | extraction §5「UB = 192KB（Arch::Ascend950 资源池）」+ 卡 §6「峰值 ~160-190KB 贴 192KB 上限，BT=128 建议降 BV=64」 | 950 按 **248KB 可用**校核：BT=128/K=128/BV=128（~190KB）**950 上可行**；「降 BV=64」是 **910b（192KB）** 的正确结论（MCTS 910b1 冠军正是 BV=64 [EXP]），被旧版误当通用结论。按平台分态 | hardware_platforms.md [SRC NPUTargetSpec.td:223-235]；MCTS opt-round-2/5 [EXP] |
| 2 | 卡 §4「一块一 program 即可……不属于 elementwise 家族的超额 block 负引导场景」 | 自我豁免无实测支撑（C-012 违规）。默认改 **persistent-core**：`grid = min(task_num, num_aicore)`，核内 `for task in range(pid, NUM_TASKS, NUM_CORE)`——对长串行递推任务零成本（每 task 本来就是长循环），且免除「task_num 远超核数时海量 program launch」风险。旧形态 + MCTS 7.23x 记录留作 P4 两态对照 | C-011 [EXP+DOC]；C-012 |
| 3 | 骨架 `kk = tl.arange(0, K)`、`vv = ... arange(0, BLOCK_V)` 直接用真实 K/V；wrapper `K=K` 原样传入 | **潜伏缺陷：官方 50-case benchmark 的 K∈{7..80}、V∈{5..50} 非 2 的幂**——`tl.arange(0, 7)` 非法，且 tl.dot 最小维 16。旧骨架在官方正确性套件上**编译即失败**（只有 MCTS 的 K=V=128 大 shape 能跑）。修正：`BK = max(16, next_pow2(K))`、`BV_TILE = max(16, min(128, next_pow2(V)))`，kk/vv 全程 mask（kmask/vmask），h 条带/initial_state/final_state 读写全部带 mask | tl.arange 2 的幂约束 + tl.dot 最小维 16（Triton 语义）；`npu_benchmark/level4/85_*.json` / kernelbench workload.jsonl case 实测 shape |

（另：旧骨架 `if output_final_state or True:` 恒分配 fs 且 STORE_FINAL 恒 True——本版改为
按 output_final_state 两态；Triton 的 h 条带寄存器常驻，**不需要** aclnn 的不可见
finalStateCompute 工作区。）

## 1. 分核策略 → grid 设计

| Ascend C 事实 | Triton 决策 | 可达性 | 依据 |
|---|---|---|---|
| 任务 = (batch, vHead) 对，chunk 维流内串行（scheduler :212-222 ✅复验） | 任务 = **(b, hv, v_block) 条带**（V 列条带独立，kernel .hpp:556,682）；chunk 维 program 内串行 | ✅ | 旧版结论维持 |
| 每核 2 流 ping-pong（arch35）；A2 单流 | 不需要——Triton 无跨核流概念 | ✅（机制不需要复刻） | — |
| blockDim = aicCoreNum（tiling_processor.h:91） | **persistent-core：grid = min(NUM_TASKS, num_aicore)**，`for task in range(pid, NUM_TASKS, NUM_CORE)`，每 task 内 h 重新初始化 | ✅ | **C-011**：CV 融合 grid=Cube 数 [DOC]；核数运行时查询、禁止写死（hardware_platforms §3：950 AIC 4~36 跨度 9 倍）。⚠️ 旧版「一块一 program」已修正（§0.6-2）；MCTS 910b 实测（一块一 program 7.23x）留作 P4 两态 |
| 变长：流内 O(tokenBatch) 重扫 cu_seqlens（scheduler :229-254） | wrapper 预计算 cu/co（torch.cumsum），kernel 标量 load（int64 张量 → 标量 int32） | ✅ 且更优（O(1)） | 4e：int64 仅限标量 load |
| 65535 逻辑核上限 | 不涉及：grid ≤ 物理核数 | ✅ | C-004 |

## 2. tilingdata → BLOCK 超参

| 字段 | Triton 对应 | 决策 |
|---|---|---|
| chunkSize ∈ {16,32,64,128}（官方小 case 含 16/32；MCTS 64/128） | `BT: tl.constexpr` = chunk_size | 全 chunk 单 tile（均为 2 的幂 ✅） |
| kHeadDim | `BK: tl.constexpr` = **max(16, next_pow2(K))** + `kmask = kk < K` | §0.6-3：非 2 幂 K 必须 padding + mask；dot 最小维 16 |
| vHeadDim ≤ 256 | `BV_TILE: tl.constexpr` = **max(16, min(128, next_pow2(V)))**，NVB = cdiv(V, BV_TILE)，`vmask = vv < V` | V=256 → NVB=2；非 2 幂 V 同法 mask |
| blockTokens = 末 chunk 尾长（scheduler :325） | 运行时 `actual = min(BT, remain)`，全程 tmask | pad 行贡献恒 0，正确性不受影响 |
| dtype 枚举三件套 | 指针 dtype 自然携带 + `USE_G/USE_GK/USE_INITIAL/STORE_FINAL/VARLEN` constexpr | 「g/gk 都不给」wrapper 拦截（tiling.cpp:91-93 同款） |
| workspace 六段 offset | **不需要**（h 条带片内常驻） | ❌ 全部消解 |

## 3. tilingkey / 模板分支 → constexpr

tilingkey 1/2（V≤128 / V=256，TileShapes128/256）→ 不进 kernel（BV_TILE/NVB 由 wrapper
定）；useG/useGk 三分支 → `USE_G/USE_GK` 两 constexpr；useExp2 恒 false + epilogue
×LN2+Exp → **`tl.exp(gk_last * LN2)`，与源码逐位同构**（regbase.hpp:53-73 ✅复验；
`tl.exp2` 是否更优 ⚠️ 未实测两态，勿断言——C-012 合规写法）。

## 4. 片上内存 → 分平台预算校核（每 program）

| tile（fp32 计） | BT=128, BK=128, BV=128 | BT=64, BK=128, BV=64 |
|---|---|---|
| h 条带 [BK, BV] 常驻 | 64KB | 32KB |
| w/k tile [BT, BK] input dtype ×2（瞬态） | 64KB | 32KB |
| u/v_new/v_up [BT, BV] ×2~3（瞬态） | 96~128KB | 48~64KB |
| **峰值** | **~190KB** | **~112KB** |
| vs **950 = 248KB** [SRC] | ✅ 富余 1.3 倍 | ✅ |
| vs **910b = 192KB** | ⚠️ 贴顶 → **BV=64（~128KB）** | ✅ |

- **配置决策**：950 → BT≤128/BK=128/BV=128 可行；910b → BV=64（与 MCTS 910b1 冠军
  BV=min(64,V) [EXP] 一致——旧版把 910b 结论误当通用，§0.6-1）。
- 双缓冲/预取由 multibuffer 承接（默认 true [SRC]；本算子 [EXP] 交叉验证：False -5.9%）；
  **不配 num_stages**（C-006）。

## 5. 流水编排 → Triton 可达手段

| Ascend C 机制 | 可达性 | Triton / 依据 |
|---|---|---|
| C1→V1→C2→V2 跨核四段流水 + CrossCoreFlag ×4 对 | ❌（机制）/✅（语义） | 单 program 顺序执行天然满足数据依赖；并行度由 (b,hv,vb) 条带补足。**结构性替代**：GM 往返逐 chunk 可数（AscendC：v_work/h_work/v_update/kDecay/h_i 读 6+ 次 vs Triton：0 次 workspace） |
| Catlass Mmad pingpong（L1/L0 tile，黑盒） | ❌ | `tl.dot`（T2 Cube），编译器管 L0/L1 |
| Reg VF 双发射（ApplyRowScale DualIssue） | ❌ | 普通 `*`/`-`/`tl.exp` 向量算子（fp32 原生 [SRC]） |
| direct-UB 直通（kernel .hpp:248-252 ✅复验） | ❌ | 不需要——比它更彻底（无 workspace GM） |
| blockTokens<16 的 AIV 纯 FMA 兜底 | ❌（不需要） | mask + padding（tl.dot 按 BT padding，贡献恒 0） |
| zN 布局 v_update | ❌ | 不需要（无 L1 显式管理） |
| AIV 开场 h[0] 初始化 + SyncAll | ✅ | 并入主 kernel：chunk i=0 先 store h |
| CV 同步编排 | ⚠️ 编译器自动 | sync_solver / auto_bind_sub_block 默认 true [SRC]；不手写 al.sync_block_* |

## 6. Compute → Triton 计算骨架（逐指令）

| Ascend C（锚点已复验） | Triton | 可达性 |
|---|---|---|
| C1: v_work = w@h（fp16 in / fp32 acc） | `tl.dot(w_t, h.to(out_dtype))` | ✅（dot 输入 = input dtype，对齐 mmad 语义） |
| V1: v_new = u − v_work | `u_t.to(tl.float32) - dot` | ✅ |
| V1: decay=exp(g_last−g)，v_up = v_new·decay | `tl.exp(g_last - g_t)`；**g_last 取 actual−1 行**（⚠️ 正确性高危：取错行静默错值） | ✅ |
| V1 双写出（v_new 原样 RTNE / v_up cast 供 C2） | `tl.store(v_new...)`；`v_up.to(out_dtype)` | ✅ |
| C2: h_work = kg^T @ v_up | `tl.dot(k_t, v_up_dt)`；k_t **转置寻址 load [BK,BT]**（禁 tl.trans，标量化 15x runbook） | ✅（⚠️ 列主序访存性能实测定） |
| V2: h·exp(g_last) 标量门 | `h *= tl.exp(g_last)` | ✅ |
| V2: h·exp2(gk_last) 逐 K 行 | `h *= tl.exp(gk_last * LN2)[:, None]`（kmask） | ✅（与源码逐位同构） |
| V2: h += h_work | `h += dot(...)` fp32 | ✅ |
| h/final_state 写出 CAST_RINT | store 隐式 RTNE；fp32 state 直通天然成立（条带本就 fp32） | ✅ |
| **寻址（4e）** | `b.to(tl.int64)` 大基址 + int32 块内偏移 | ✅（mhc_pre 同款纪律） |

### 不可达项汇总

1. ❌ Catlass Mmad / L1 管理 → tl.dot（黑盒依赖出仓，manifest 缺省项 4 仍成立）
2. ❌ CrossCoreFlag 四段 CV 流水 / 双流 ping-pong → 单 program 顺序 + 条带并行（结构性替代）
3. ❌ Reg VF 双发射 / direct-UB / zN / ComputeTail → 片内常驻条带天然覆盖或不需要
4. ⚠️ 转置寻址 load [BK,BT] 性能 → 实测定；劣化则 wrapper 预转置（未实测勿断言）

## 7. 精度策略 → 标杆与验收

- 中间域全 fp32（h 条带、v_new、decay）；dot 输入 cast input dtype（对齐 mmad 语义）——
  全 fp32 dot 为 ⚠️ 未实测两态。
- 标杆：`forward_h_trans_cpu`（test_fwd_h.py:111-115 ✅复验）全 fp32 torch 递推。
- 验收对齐官方比较器（data_compare_h.py:133,195-205）：**rel diff 1e-4、≥99.99% 元素达标、
  单点 max_diff < 0.1**，h/v_new/final_state 三输出分别判定。gk/both 路径 golden 未覆盖
  （g-only），KDA 公式以 README+epilogue 源码为准自建标杆。
- **填充纪律**：g 必须 chunk 内 cumsum 单调负衰减（base∈[−15,−5]+U(−2,0)，
  gen_decay_data:246-262）；无结构随机 g → exp(g_last−g_t) 指数爆炸出容差域。
- **benchmark 双套件**：正确性 = 官方 50 case（`npu_benchmark/level4/85_*` ≈ kernelbench
  workload，小 shape、K/V 非 2 幂、gate_mode 0/1/2、initial_state/final_state/state_v_first
  组合）——**§0.6-3 的 mask 泛化是跑通该套件的前提**；性能 = MCTS 8 大 shape
  （K=V=128、T 至 32768、varlen、GQA，mc.json）。

## 8. 显式丢弃 / 不涉及清单

| 项 | 处理 | 理由 |
|---|---|---|
| GM workspace 六段（vWork/vUpdate/kDecay/hWork/numSeq/numChunks） | 丢弃 | h 条带片内常驻，v_up 片内直供 dot（结构性更优，GM 往返可数论证） |
| CrossCoreFlag 4 对 + EVENT_ID0-3 事件组 | 丢弃 | 无跨核协作 |
| ComputeTail 逐行 FMA 兜底 | 丢弃 | mask 取代 |
| aclnn TransposeLastTwo（state_v_first） | wrapper `transpose(-1,-2)` 零拷贝 | stride 交换即可 |
| aclnn 不可见 finalStateCompute（output_final_state=false） | 不需要 | 条带寄存器常驻，无中间落地 |
| A2 单流差异 / UB 偏移 8K 差 | 不涉及 | Triton 无此分叉 |

## P2 质量门自查

- extraction 全部事实有去向：转化 10 条、结构性替代 5 条、显式丢弃 6 类 ✅
- 六任务检索全部命中 backend_facts 首层 ✅
- 旧版 2 处错误论断 + 1 处骨架潜伏缺陷已修正（§0.6）；所有「等价/豁免」断言或改结构性
  论证、或标 ⚠️ 未实测两态（C-012 合规）✅
- 骨架 preflight 12+1 项 0 真命中（记录见 triton_skeleton.py 头注）✅
