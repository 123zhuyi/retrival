# P1 语义抽取 — chunk_gated_delta_rule_fwd_h

> 只搬事实，不做转化。锚点格式 `file:line`，文件根 = `src/chunk_gated_delta_rule_fwd_h/`。
> arch35（Ascend950）为主线；A2（AtlasA2）差异单独标注。

## 1.1 功能分析（算什么）

**一句话**：GDN/KDA 的 chunk 间状态递推前向——给定逐 chunk 的 k/w/u/g(/gk)，递推产生每个 chunk 的起始状态 h[B,HV,Nc,K,V]、修正后的 v_new[B,HV,T,V]，以及可选的最终状态 final_state[N,HV,K,V]（README.md:3-14）。

**数学公式**（README.md:20-27 与 golden `forward_h_trans_cpu` tests/pta/test_fwd_h.py:111-115 一致；对每个 (batch n, v-head h, chunk i)）：

```
v_new[t]   = u[t] − w[t] @ h_i                    # t ∈ chunk i 内的 token 行
decay[t]   = exp(g[last] − g[t])                  # g = 标量门（自然对数域，chunk 内 cumsum 语义）
decay_k[k] = exp2(gk[last, k])                    # gk = 逐 K 门（log2 域）
h_{i+1}    = h_i · exp(g[last]) · decay_k[:]  +  kg^T @ (v_new · decay[:,None])
```

- `kg`：useGk 时由**调用方预衰减**（`kg = k · exp2(gk_last − gk)`，epilogue 注释 block_epilogue_gdn_fwdh_vnew.hpp:407「KDA passes kg = k * exp2(g_last - gk). Keep that decay exactly once.」），kernel 内不再对 k 乘 decay，只做 k→kDecayWorkspace 的搬运；useG-only 时 kg = k 本体。
- **g 是 chunk 内累积量**：数据生成器 `gen_decay_data` 对每个 chunk 段做 `cumsum(0)`（test_fwd_h.py:261），故 `g[last]` 是整个 chunk 的总对数衰减，`g[last]−g[t]` 是 t→chunk 末的剩余衰减。gk 同语义（log2 域）。
- 末 chunk 的 h_{i+1} 写入 `final_state` 而非 h（golden :113-115；kernel 侧 `NeedProcessStage2 = storeFinalState || !isFinalState` block_scheduler_gdn_fwd_h.hpp:376-378）。
- `h[i]` 存的是 **chunk i 的起始状态**（反向要用），故 chunk0 的 h[0] = initial_state（或 0），由 AIV 在主流水前先初始化（gdn_fwd_h_kernel.hpp:814-900）。
- 反向算子：本目录无（bwd 在同仓其他目录，未下载）。

## 2 原型设计（接口）

8 输入 3 输出 7 属性（op_host/chunk_gated_delta_rule_fwd_h_def.cpp:42-103）：

| 名称 | 必选性 | dtype | shape（aclnn CheckShape, aclnn_chunk_gated_delta_rule_fwd_h.cpp:112-169） |
|---|---|---|---|
| k | 必选 | fp16/bf16 | [B, H, T, K]，head-major BNSD |
| w | 必选 | 同 k | [B, HV, T, K]（与 k 同 B/T/K） |
| u | 必选 | 同 k | [B, HV, T, V] |
| g | 可选 | fp32 或同 k | [B, HV, T] |
| gk | 可选 | fp32 或同 k | [B, HV, T, K]（与 k 同 B/T/K，HV 同 u） |
| initial_state | 可选 | fp16/bf16/fp32 | [N, HV, K, V] |
| cu_seqlens | 可选（ValueDepend） | int64 | [tokenBatch+1]，变长模式 |
| chunk_indices | 可选（ValueDepend） | int64 | 变长模式（kernel 未实际使用，仅透传） |
| h | 必选输出 | 同 k | [B, HV, Nc, K, V] |
| v_new | 必选输出 | 同 k | [B, HV, T, V] |
| final_state | 可选输出 | 同 initial_state（无 initial_state 时 fp32） | [N, HV, K, V] |

属性（def.cpp:96-103）：`output_final_state`(bool)、`chunk_size`(int)、`logical_batch/seqlen/k_heads/v_heads/k_dim/v_dim`（6 个 int，由 l0op 从 shape 现场推导填入，op_api/chunk_gated_delta_rule_fwd_h.cpp:60-72）。

aclnn 层另有两个**接口层语义**（不进 kernel）：
- `state_v_first=true`：h/final_state/initial_state 的末两维 [K,V]↔[V,K] 由 aclnn 的 `TransposeLastTwo` + ViewCopy 消化（aclnn_*.cpp:83-94,321-366），kernel 永远只见 [K,V]。
- `output_final_state=false`：aclnn 内部分配不可见 finalStateCompute（dtype = initial_state 或 fp32，aclnn_*.cpp:330-336），kernel 的 storeFinalState 仍按 attr 走。

## 2.1 原型约束

- **g/gk 至少给一个**（tiling.cpp:91-93；aclnn :229-234）；都给时 dtype 必须相同（aclnn :183-186）。
- `HV ≥ H 且 HV % H == 0`（aclnn :124-125；tiling.cpp:130-139）→ GQA 式头分组，`headGroups = vNumHead / kNumHead`（block_scheduler_gdn_fwd_h.hpp:213），`kHeadIdx = vHeadIdx / headGroups`（:267）。
- `vHeadDim ≤ 256`（tiling.cpp:140-144）；README 声明 V∈{128,256}、chunk_size∈{64,128}（README.md:89）。
- k/w/u 同 dtype（fp16/bf16）；h/v_new 同 k dtype；gate 为 fp32 或与 k 同 dtype；state fp16/bf16/fp32（aclnn CheckDtype :171-196）。
- 变长（cu_seqlens 非空）时：shapeBatch=1、tokenBatch=cu_seqlens.len−1（tiling_processor.h:79-89）；空序列（len=0）被跳过（scheduler InitRuntime :190-199）。
- 所有输入 AutoContiguous（def.cpp:45 等）。

## 3 规格约束

| 项 | 值 | 锚点 |
|---|---|---|
| 芯片 | ascend910b / ascend910_93 / ascend950（A2/A3/A5） | def.cpp:116-118；README.md:87 |
| kernel 类型 | **MIX_AIC_1_2**（CV 融合，1 Cube : 2 Vector） | chunk_gated_delta_rule_fwd_h.cpp:119,123 |
| 数据路径 | k/w/u/h/v_new: fp16/bf16；gate: fp16/bf16/fp32；state: fp16/bf16/fp32 | def.cpp:23-37 |
| format | 全 ND，AutoContiguous | def.cpp:38-40 |
| 动态 shape | DynamicCompileStatic/DynamicRank/DynamicShape 全 true | def.cpp:106-114 |
| 交付 | aclnn 两段式 + PTA `torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h` | tests/pta/test_fwd_h.py:323 |

## 4.1.1 分核策略

- `blockDim = aicCoreNum`（AIC 核数，tiling_processor.h:91）——**CV 融合下 AIC 数即 grid**；AIV 侧 `GetBlockIdx()/GetSubBlockNum()` 折算回逻辑核（block_scheduler_gdn_fwd_h.hpp:405-408）。
- **任务单元 = (batch, vHead) 对**，`taskNum = batch × vNumHead`（scheduler :212）；chunk 维**不并行**——流内 chunk 递推天然串行（h 依赖）。
- 每核领 2 条流 ping-pong：`maxTaskCntPerLoop = taskNum > coreNum ? 2 : 1`，`taskStride = coreNum × maxTaskCntPerLoop`，流 s 的任务序列 = `coreIdx×2+s, +stride…`（:214-222）。arch35 才双流；**A2 固定单流**（注释：短尾 chunk 会让一个 AIV 子核先退役，双流协议无反向信用，gemm/block/block_scheduler_gdn_fwd_h.hpp:214-216）。
- 变长模式：先扫 cu_seqlens 得实际 batch（跳过 0 长序列）与 chunk 前缀和（:185-203）；每条流 `ResolveVarlenSequence` 重扫定位（:229-254，**O(tokenBatch) 逐流重扫，无缓存**）。
- 任务 offsets（`UpdateTask` :300-330）：
  - `hSrcOffset = ((b·HV + hv)·totalChunks + chunkOffset + i)·K·V`；`hDstOffset = hSrcOffset + K·V`；**末 chunk 且 storeFinalState 时 hDst=hSrc（原地）**（:312-314）；
  - `uvOffset`（u/v_new 共用）、`wkOffset`（k/w 的 K-head 基址，w 用 vHead 行、K-head 列——w 的 head 维是 HV）、`gOffset`、`gkOffset`；
  - workspace 按 `(cubeCoreIdx×2 + streamId)` 分段：hWork K×vBlock、vWork chunk×vBlock、kDecayWork chunk×K（:320-322）；
  - `blockTokens = 末 chunk ? batchTokens − i·chunkSize : chunkSize`（:325）。

## 4.1.2 单核内切分

- Cube mmad tile：`GDNFwdHTileShapes128{L1 128×128×128, L0 同}`（V≤128）/ `GDNFwdHTileShapes256{L1 128×256×128, L0 128×256×64}`（V=256）（gdn_fwd_h_kernel.hpp:60-68）。**tilingKey = vHeadDim>128 ? 2 : 1**（tiling.cpp:152-154）。
- C1 `v_work = w@h`：形状 [blockTokens, V] ← [blockTokens, K]×[K, V]，fp32 累加（WORKSPACE_TYPE=float，kernel .cpp:48）。
- C2 `h_work = kg^T @ v_update`：形状 [K, V] ← [K, blockTokens]×[blockTokens, V]；v_update 以 **zN 布局**存放（VUpdateType layout::zN，kernel .hpp:110）供 L1 直取。
- `blockTokens < 16` 的 chunk **Cube 跳过 mmad**（kernel .hpp:547-551,670-674），dense 模式由 AIV 向量兜底（ComputeTailV/HWorkspace，逐行 FMA，:353-506）；useBoundedMmad（变长或 T%chunk≠0，:521）走 `BlockMmad{Tla}Tail`（DispatchPolicyTlaTail 单 stage 有界 mmad）。
- AIV 侧 epilogue 行切分：两子核对半分 M 行（`CeilDiv(mActual, 2)`），ROW_TILE=16 逐 tile（vnew epilogue :268,276-281；update epilogue :222,228-233）。

## 4.1.3 tilingdata / tilingkey

20 字段（struct.h:28-52 = tiling.h:23-47 宏版）：shape 7（batch/seqlen/kNumHead/vNumHead/kHeadDim/vHeadDim/chunkSize）+ 开关 4（useInitialState/storeFinalState/useG/useGk）+ dtype 枚举 3（dataType/gDataType/stateDataType，**约定 0=fp16/1=bf16/2=fp32**，tiling_processor.h:31-34；kernel 侧只用 fp32 判定与 bf16 判定，:70-72,97-98）+ 变长三元（isVariedLen/shapeBatch/tokenBatch）+ workspace offset 6。

workspace 布局（tiling_processor.h:97-121，均以 512B 对齐，前后各垫 16MB RSVD）：
| 段 | 大小 |
|---|---|
| vWorkspace（C1 输出，fp32） | cores·chunk·V·4B·2 |
| vUpdateWorkspace（衰减后 v_new，input dtype，zN） | cores·chunk·V·elemsize·2 |
| kDecayWorkspace（仅 useGk） | cores·chunk·K·elemsize·2 |
| hWorkspace（C2 输出，fp32） | cores·K·V·4B·2 |
| numSeq / numChunks | (tokenBatch+1)·8B 各一 |

tilingkey：1 = V≤128（TileShapes128），2 = V=256（TileShapes256）；MIX_AIC_1_2 随 key 注册（kernel .cpp:118-126）。

## 5 片上内存（UB 预算，arch35）

UB = 192KB（Arch::Ascend950 资源池）。各消费者偏移硬编码：

| 消费者 | 偏移 | 用途 |
|---|---|---|
| 共用 | 0 | `calcUbTensor` fp32 计算缓冲（epilogue 复用） |
| V1/V2 ping | 32K | hUpdate / v_work 接收（fp32，chunk×V 或 K×V） |
| V1 wide IO | 16K/24K | nv>128（V=256）时的 u/v_new 宽行 ping/pong（vnew epilogue :72-73,91-100） |
| kernel 主 | 32K/96K | ubVWorkPing/Pong（direct-UB 模式 C1 直写）、ubHUpdatePing/Pong（C2 直写）（gdn_fwd_h_kernel.hpp:272-275） |
| h 初始化 | 0/96K state + 64K/160K h | AIV 开场初始化 h[0] 的 ping-pong（:835-842）；`rowsPerTile = min(32K,64K)/(V·elemsize)`（:827-830） |
| V1 g 系 | 160K-165K | g/gLast/gInput ping-pong 各 1KB（vnew :74-79） |
| V1 share | 166K | gBrcb（未实际使用，:102） |
| V2 update 专用 | 160K scratch + 176K gLast + 168-173K gk 系 | update epilogue :67-74 |
| tail 兜底 | 166K-170K | ComputeTail 五区（input/float/accum/weightInput/weightFloat，kernel :367-371） |

L1：mmad 由 Catlass 管理（L1TileShape 128×128×128 / 128×256×128）；另有 `l1VUpdatePing/Pong` 承接 V1 写出的 zN v_update（0 与 chunk×V·2B 处，:277-278）。

## 6 流水编排（CV 协同协议）

**宏观**：`SyncAll` 开场（可作 megakernel 后续 stage，kernel :509-511）→ AIV 先初始化全部 h[0]（:814-900）→ `SyncAll` → C1/V1/C2/V2 四段流水线随 chunk 递推滚动。

**四段**（每个 (b,h) 流的每个 chunk）：
1. **C1（AIC）**：`v_work = w @ h_i` → fp32 GM workspace（或 direct-UB 模式直写 UB）。等 `vec2Done[stream]`（上一 chunk 的 V2 已读完 h_i）。`CrossCoreSetFlag cube1Done`（:619,651）。
2. **V1（AIV）**：等 cube1Done；`v_new = u − v_work`（fp32）；标量门时乘 `decay[t]=exp(g_last−g_t)`（PrepareG + ApplyRowScale regbase VF）；**双写出**：v_new（未衰减，回 GM 输出）+ v_update（衰减后 cast fp16/bf16 → L1 zN → C2）；useGk 时把 k 拷入 kDecayWorkspace；`SetFlag vec1Done`。
3. **C2（AIC）**：等 vec1Done；`h_work = kg^T @ v_update`（kg=k 或 kDecayWorkspace）→ fp32 hWorkspace；末 chunk 且不存 final_state 时跳过（NeedProcessStage2=false）；`SetFlag cube2Done`。
4. **V2（AIV）**：等 cube2Done；读 h_i（或 initial_state/final_state 直通），乘 `exp(g_last)`（标量门，单值 Exp :261-283）与 `exp2(gk_last)`（逐 K 行，PrepareKGate+ApplyRowScale :350-376），`+ h_work`（fp32 Add :388-390）；写 h[i+1]（CAST_RINT 回 input dtype）或 final_state（**fp32 直通路径**：`useFp32StateUpdate` 时中间 chunk 也经 finalState GM 保持 fp32，避免逐 chunk 精度损失，:258-259,399-421）；`SetFlag vec2Done`。

**同步原语**：CrossCoreFlag 4 对（cube1Done/vec1Done/cube2Done/vec2Done = flag id 0-7，scheduler :125-128）+ AIV 内部 HardEvent 事件组（EVENT_ID0-3 × ping/pong 偏移 4）+ direct-UB 模式的 FREE/READY flag（id 1-2/6-7，kernel :164-168）。

**direct-UB 直通**（`useDirectFp32Ub`，kernel :248-252）：全条件 = workspace fp32 + 非变长 + chunk≤64 + T%chunk==0 + K==V==128 + task 数 ≥ 核数 → C1/C2 的 mmad 结果**直写 UB**（PackedTileCopyTlaToUB，NO_SPLIT），AIV 从 UB 读，省 GM 往返。

## 6.x 计算流程（与 golden 逐行对照）

| 步 | golden（test_fwd_h.py:111-115） | kernel 实现 | 精度域 |
|---|---|---|---|
| ① h[0] 初始化 | `S[...,0] = initial_state`（i==0） | AIV 开场拷贝或清零（:867-893）；state≠input dtype 时 Cast RINT（:875-877） | fp32→input dtype 单cast |
| ② v_new = u − w@h_i | :111 `u_sel − w_sel @ S[...]` | C1 mmad（fp32 累加）→ V1 Sub | w@h fp32；减在 fp32 |
| ③ 衰减 v_update = v_new·exp(g_last−g) | :113 `(v_new * (g[last]−g).exp()[...,None])` | V1 PrepareG（sub+exp，useExp2 时 ×LN2）+ ApplyRowScale | fp32 |
| ④ h 衰减基 = h_i·exp(g_last)·exp2(gk_last) | :113 `S·g[last].exp()`（golden 仅标量门） | V2 Muls(muls) + ApplyRowScale(gkLast) | fp32 |
| ⑤ h_{i+1} = ④ + kg^T@v_update | :113 `+ k_sel^T @ (...)` | C2 mmad（fp32）→ V2 Add | fp32 |
| ⑥ 写回 | `S.to(dtype_)` / final_state fp32 | V2 Cast RINT 写 h；final_state fp32 直通 | 单次 RINT |

**kGated 特别路径**：kg 预衰减由调用方完成；kernel 的 kDecayWorkspace 只是 k 的布局搬运（vnew epilogue :404-418）。**gk 行衰减在 V2 作用于 h_i 的 K 行**（gkLast[k]，:350-376）——注意这与 h_next 公式中 `decay_k` 乘在 h_prev 一致。

## 精度策略

- **全中间域 fp32**：mmad 累加 fp32（WORKSPACE_TYPE=float）、workspace fp32、epilogue 计算 fp32；仅写出 h/v_new/v_update 时 Cast（`CAST_RINT` = RTNE，vnew :374,396；update :402,423,437,444）。
- state dtype ≠ input dtype 时（如 fp16 输入 + fp32 state）：`useFp32StateUpdate` 让 h 递推全程在 fp32 GM（finalState）上进行，中间 chunk 不降级（update epilogue :258-259,399-421；注释 :1000-1002「Update always writes the FP32 state through MTE3, including intermediate chunks.」）。
- 初始 state→h[0] 的 Cast 用 RINT（kernel :876）。
- gate 输入 fp16/bf16 时先 Cast 到 fp32 再 exp（vnew PrepareG :185-188；regbase b16→f32 unpack zeroing cast，regbase.hpp:20-40）。
- **exp2 实现 = ×LN2 + Exp**（ PrepareKGateRegbase<T,true> regbase.hpp:53-73；vnew PrepareG :202-205；update :275-278）——log2 域约定由 README 固定（README.md:29-30），kernel 模板 useExp2 恒 false，真正的 exp2 语义全在 epilogue 的 ×LN2 里。

## 测试设计

- **标杆**：单标杆 `forward_h_trans_cpu`（CPU fp32 全精度递推，test_fwd_h.py:39-138），输出三件套与 kernel 对齐。
- **比较器**：`data_compare`（data_compare_h.py:133-181）——相对误差阈值 `diff_thd=1e-4`、达标率 `pct_thd=1e-4`（即 ≥99.99% 元素达标）、单点 max_diff 硬上限 0.1（:195-205 调用处）。
- **数据生成**：k/w/u ~ N(0,1)（gen_input_data :267-272）；g = base∈[−15,−5] + U(−2,0) 后 chunk 内 cumsum（:246-262，保证衰减单调且 chunk 末值为总衰减）；initial_state ~ N(0,1) fp32。
- **典型 shape**：注释给出 HQ=HK=16、HV=32（:52）；命令行 17 参数（batch/seqlen/heads/dims/varlen/chunk/dtype/设备号等，:151-169）。
- **入口**：`torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h`（:323）——PTA 注册路径，g/gk 至少其一（:322 注释）。
- 变长：`gen_seqlen` 随机切分（:221-230）；`chunk_indices` 实际传 zeros 占位（:244）。
- 无官方 ST json / UT tiling 用例（非 CANN 仓结构）——缺省项。

## 架构差异（A2 副线）

| 点 | arch35（主线） | A2（op_kernel/ 顶层） |
|---|---|---|
| ArchTag | Arch::Ascend950 | Arch::AtlasA2（policies.hpp:18） |
| scheduler 流数 | taskNum>核数时 2 流 ping-pong | 固定 1 流（注释：短尾 chunk 非对称退役无反向信用） |
| update epilogue | 含 useFp32StateUpdate 直通 + regbase include | 无 fp32 直通（少该分支） |
| UB 偏移 | g 系 168K+（vnew 侧 160K+） | g 系 160K+（偏移整体低 8K） |
| tail 兜底 | 同有 | 同有（ComputeTail 在公共 kernel 侧） |
