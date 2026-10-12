# P2 语义转化 — chunk_gated_delta_rule_fwd_h（Ascend C → Triton）

> 输入：extraction.md。映射规则引用 [`ascendc_triton_semantic_mapping.md`](../../ascendc_triton_semantic_mapping.md)。
> 可达性三态 + ⚠️ 严重性分级（正确性风险/性能风险）（README v1.3）。引用 Triton 上游特性处已对照后端契约 runbook 卡（2026-09-09 批次）。

## 1. 分核策略 → grid 设计

| Ascend C 事实 | Triton 决策 | 可达性 | 依据 |
|---|---|---|---|
| 任务 = (batch, vHead) 对，taskNum = batch·vNumHead，chunk 维串行递推（scheduler :212-222） | **grid = (B·HV·NVB,)**，NVB = V/BLOCK_V 的 V 向分块数；每 program 一条 (b, hv, v_block) 状态条带，内部 for chunk 串行 | ✅ | scheduler UpdateTask offsets 结构 |
| 每核 2 流 ping-pong（arch35），A2 单流 | 不需要——Triton 无跨核流概念；**(b,hv,v_block) 条带数 ≫ 核数时由 persistent 循环自然摊销** | ✅（机制不需要复刻） |  |
| **关键观察：V 维完全并行**。h 的 K×V 中，v_block 列条带的递推只依赖 u/v_new 的同一 v_block（C1 的 w@h 按列块独立、C2 的 kg^T@v_up 按列块独立） | h 条带 [K, BLOCK_V] 常驻寄存器/UB，整轮递推**不出 program** | ✅ | kernel C1/C2 GemmCoord 形状（kernel .hpp:556,682）按 n=vBlockDim 切 |
| V=256 时 L1 tile 128×256（TileShapes256，tilingkey=2） | BLOCK_V=128 + NVB=2（单 program fp32 h 条带 K×256=128KB 超 UB 预算；两条带独立 program） | ✅ | UB 192KB 预算校核见 §4 |
| 变长：流内 O(tokenBatch) 重扫 cu_seqlens 定位（scheduler :229-254） | wrapper 预计算 `cu_seqlens`/`chunk_offsets`（torch.cumsum），kernel 标量 load 两个值；grid = N·HV·NVB | ✅ 且更优（O(1) vs O(N) 重扫） | test_fwd_h.py:33-37 同款预计算 |

## 2. tilingdata → BLOCK 超参

| 字段 | Triton 对应 | 决策 |
|---|---|---|
| chunkSize ∈ {64,128} | `BT: tl.constexpr` = chunk_size | 全 chunk 单 tile（BT≤128，对齐 2 的幂） |
| kHeadDim（K=128 主线） | `BK: tl.constexpr` = next_pow2(K) | C1/C2 的 K 维 |
| vHeadDim ∈ {128,256} | `BLOCK_V: tl.constexpr` = min(next_pow2(V), 128)，NVB = V // BLOCK_V | V=256 拆 2 program |
| blockTokens = 末 chunk 尾长 | 运行时标量 `actual = min(BT, T_chunk_remain)`，全程 mask | tl.dot 的 M/K 维按 BT padding、mask 置 0（尾部贡献恒 0，正确性不受影响） |
| dtype 枚举三件套 | kernel 内不需要——wrapper 按张量 dtype 分派 constexpr（USE_G/USE_GK/STORE_FINAL/USE_INITIAL），dtype 由指针类型自然携带 | — |

## 3. tilingkey 模板分支 → constexpr

| 分支 | Triton |
|---|---|
| tilingkey 1/2（V≤128 / V=256） | 不进 kernel——BLOCK_V/NVB 由 wrapper 计算，kernel 单一形态 |
| useGk/useG 三分支（both / gk-only / g-only） | `USE_G` / `USE_GK` 两个 constexpr，4 组合中「都不给」由 wrapper 拦截（tiling.cpp:91-93 同款校验） |
| useExp2 恒 false + epilogue ×LN2 实现 exp2 | **语义结论**：gk 在 log2 域 → `tl.exp(gk_last * LN2)`；g 在自然对数域 → `tl.exp(...)`。⚠️ 未验证项：Triton-Ascend 是否有高效 `tl.exp2`——**按 README v1.3 规范不做无实测断言**，骨架统一用 `exp(x·ln2)`（与源码逐位同构），如需 exp2 自行实测 |

## 4. 片上内存 → 预算校核（每 program，BLOCK_V=128、K=128、BT=128 峰值）

| tile | 字节 |
|---|---|
| h 条带 fp32 [128, 128] | 64KB（常驻） |
| w [BT,K] fp16 + k [BT,K] fp16（转置 load） | 32+32KB（瞬态） |
| u/v_new/v_up [BT,BV] | 32KB×2~3（瞬态） |
| **峰值** | ~160-190KB，贴 UB 192KB 上限——**BT=128 且 K=128 时建议 BLOCK_V=64 或 BT 内部分两段**；BT=64/K=128 时 ~120KB 安全 |

⚠️ 性能风险：Triton 无显式双缓冲控制（num_stages 在 Triton-Ascend **运行时不生效**，见 runbooks/autotuning/triton-ascend-microkernel-autotune-noise.md）——流水掩盖交给编译器，超预算时优先降 BLOCK_V。

## 5. 流水编排 → Triton 可达手段

| Ascend C 机制 | 可达性 | Triton / 降级 |
|---|---|---|
| C1→V1→C2→V2 跨核四段流水 + CrossCoreFlag 握手 | ❌（机制）/✅（语义） | **单 program 内顺序执行天然满足数据依赖**；跨核流水是 Cube/Vector 异构的资源利用手段，Triton 统一编程模型下无需复刻。预期性能差距：Cube 利用率——Triton tl.dot 也会走 Cube，但少了与 V 核的显式交叠 |
| Catlass Mmad pingpong（L1/L0 tile、PackedTileCopyTla） | ❌ 黑盒（依赖出仓） | `tl.dot`，编译器自动管 L0/L1 |
| Reg 级 VF_CALL（ApplyRowScaleDualIssue / ComputeVNewRegbaseDualIssue 双发射） | ❌ | 普通 `*`/`-` 向量算子；双发射由编译器调度 |
| direct-UB 直通（C1/C2 直写 UB 省 GM 往返） | ❌ | 不需要——Triton 的 h 条带本来就在片内，**比 direct-UB 更彻底**（连 workspace GM 都不存在） |
| blockTokens<16 的 AIV 纯 FMA 兜底 | ❌（不需要） | mask + padding 即可（tl.dot 最小粒度由编译器处理） |
| zN 布局 v_update 供 L1 直取 | ❌ | 不需要（无 L1 显式管理） |
| AIV 开场 h[0] 初始化 + SyncAll | ✅ | 并入主 kernel：chunk i=0 时先存 h（initial_state 或 0） |

## 6. Compute → Triton 骨架（逐指令）

| Ascend C | Triton | 备注 |
|---|---|---|
| C1: v_work = w@h（fp16 in / fp32 acc） | `tl.dot(w_t, h_dt)`，h_dt = `h.to(dtype)` | **精度对齐源码**：dot 输入为 input dtype（对应 mmad fp16 输入），非全 fp32 |
| V1: v_new = u − v_work | `u − dot`（fp32） | |
| V1: decay=exp(g_last−g)，v_up = v_new·decay[:,None] | `tl.exp(g_last − g)`，`g_last` 标量 load（**取 actual−1 行**，尾 chunk 不是 BT−1） | ⚠️ 正确性风险：取错行静默错值 |
| V1 双写出（v_new 原样 / v_up cast RINT） | v_new store（隐式 cast）；`v_up_dt = v_up.to(dtype)` 供 C2 | |
| C2: h_work = kg^T @ v_up（kg：useGk 时=预衰减 k，否则=k） | `tl.dot(k_T, v_up_dt)`，k_T 用**转置寻址直接 load [BK, BT]** | ⚠️ 性能风险：`tl.trans` 在 Triton-Ascend 标量化 ~15x（runbooks/compute-pipeline/triton-ascend-trans-scalarization.md），**禁用 tl.trans，改用转置寻址 load**（列主序访问，stride=K 的离散列——可参考 shifted-load 卡评估，实测定） |
| V2: h·exp(g_last)（标量门） | `h *= tl.exp(g_last)` | 标量 |
| V2: h·exp2(gk_last) 逐 K 行（k 门） | `h *= tl.exp(gk_last * LN2)[:, None]`，gk_last = gk[actual−1, :] 行向量 | |
| V2: h += h_work | `h += dot(...)` | fp32 |
| h/final_state 写出（CAST_RINT） | store 隐式 RTNE | fp32 state 直通：h 条带本就 fp32，store final_state 不降级 ✅（比 AscendC 的 GM 直通更自然） |

## 7. 精度策略 → 标杆与验收

- 中间域全 fp32（h 条带、v_new、decay 均 fp32）；dot 输入 cast 到 input dtype（对齐 mmad 语义）——若 benchmark 容差更宽可试全 fp32 dot（⚠️ 未实测，两态给出）。
- 标杆：复刻 `forward_h_trans_cpu`（全 fp32 torch，chunk 内 cumsum g）。验收对齐官方比较器：**rel diff 1e-4、≥99.99% 元素达标、单点 max_diff < 0.1**（data_compare_h.py:133,195-205）。
- 填充满贯：g 用 chunk 内 cumsum 的负值衰减（base∈[−15,−5]+U(−2,0)），**不要用无结构随机 g**——decay=exp(g_last−g_t) 在非累积 g 下会指数爆炸，超出容差判定。

## 8. ❌ 不可达清单（降级已给）

1. Catlass Mmad/L1 管理（黑盒依赖出仓）→ tl.dot
2. CrossCoreFlag 四段 CV 流水 → 单 program 顺序 + (b,hv,v_block) 并行
3. Reg VF 双发射 → 普通向量算子
4. direct-UB / zN / 显式 ping-pong → 片内常驻条带天然覆盖
5. A2 单流差异 → 不适用（Triton 无此分叉）

## 9. 勿复刻清单

- **workspace 六段布局**（vWork/vUpdate/kDecay/hWork/numSeq/numChunks）：Triton 不需要任何 GM workspace——h 条带片内常驻、v_up 片内直供 dot。⚠️ 这是结构性更优，非等价断言（GM 往返次数：AscendC 每 chunk 6+ 次 vs Triton 2 次 store+必要 load）。
- 4 对 CrossCoreFlag + EVENT_ID0-3 ping-pong 事件组。
- tail 的 ComputeTail 逐行 FMA 兜底（mask 取代）。
- stateVFirst 的 aclnn TransposeLastTwo：Triton wrapper 直接交换 h/final_state 末两维 stride（零拷贝），kernel 无分支。
