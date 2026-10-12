# P0 manifest — chunk_gated_delta_rule_fwd_h

> 日期：2026-09-09 ｜ 来源仓：`github.com/jiangdm2024/flash-linear-attention-npu_dev`（天津大学版权，非 CANN 官方仓）
> 快照 pin：分支 `main` @ `4c78653f47beda77681aa725c7fa49c7ce8d86a9`（2026-08-31）
> 本地快照：`zmm_source_retrieval/src/chunk_gated_delta_rule_fwd_h/`（27 文件，仅下载该算子子目录，非全仓）

## 0. 算子定位与复杂度预判

- **算子**：ChunkGatedDeltaRuleFwdH —— GDN/KDA 共享的 **chunk 间状态递推**（recurrent state propagation）：
  `v_new = u - w @ h_prev`；`h_next = h_prev · exp(g_last) · exp2(gk_last) + kg^T @ (v_new · exp(g_last − g))`。
- **kernel 类型判定**：`KERNEL_TASK_TYPE(..., KERNEL_TYPE_MIX_AIC_1_2)`（op_kernel/chunk_gated_delta_rule_fwd_h.cpp:119,123）→ **CV 融合（1 AIC : 2 AIV）**，基于 **Catlass 模板库**（`Catlass::Gemm::Kernel::GDNFwdHKernel`），非手写 Init/Process/CopyIn 三段式。
- **复杂度预判：高价值**（不满足任一「低价值」条件）：CV 融合 + 跨核 flag 握手流水 + 状态递推串行依赖 + Reg 级 VF_CALL（`kernel_utils/vector/regbase.hpp`）+ 变长（cu_seqlens）+ 三种 dtype 域（input fp16/bf16、gate fp16/bf16/fp32、state fp16/bf16/fp32）。预期 ❌ 不可达项多（Catlass TLA/Mmad pingpong、CrossCoreFlag、Reg VF），卡片价值密度高。
- **架构二元**：`op_kernel/{gemm,epilogue}/`（顶层）= **AtlasA2**（`Arch::AtlasA2`，epilogue_policies.hpp:18）；`op_kernel/arch35/` = **Ascend950**（`Arch::Ascend950`）。按工作流约定 **arch35 为主线**，A2 差异记「架构差异」节。kernel 入口按 `__CCE_AICORE__ == 310` 选头文件（chunk_gated_delta_rule_fwd_h.cpp:15-19）。两变体核心差异：arch35 的 update epilogue 多了 `useFp32StateUpdate` 直通路径与 regbase include；scheduler 在 arch35 支持 `maxTaskCntPerLoop = PING_PONG_STAGES` 双流（A2 固定 1 流，注释称短尾 chunk 的非对称退役问题）。

## 1. 材料清单

| 类别 | 路径 | 存在 | 关键符号 / 行号 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/chunk_gated_delta_rule_fwd_h_def.cpp` | ✅ | `OP_ADD(ChunkGatedDeltaRuleFwdH)` :123；dtype 三组表 :23-37；8 输入 3 输出 7 属性 :42-103；AICore config 910b/910_93/950 :116-118 | 原型设计、规格约束 |
| aclnn 接口 | `op_host/op_api/aclnn_chunk_gated_delta_rule_fwd_h.cpp/.h` | ✅ | CheckShape :112-169（BNSD rank4、HV%H==0、h [B,HV,Nc,K,V]）；CheckDtype :171-196（gate=fp32 或与 k 同 dtype）；**stateVFirst 由 aclnn 层 TransposeLastTwo 消化** :83-94,321-354；`output_final_state=false` 时 aclnn 内部分配 finalStateCompute :330-336 | 对外接口、属性消化策略 |
| l0op 注册 | `op_host/op_api/chunk_gated_delta_rule_fwd_h.cpp` | ✅ | logical_* 7 属性从 shape 现场推导 :60-72；cu_seqlens/chunk_indices 转 int64 tensor :40-57 | aclnn↔GEIR 属性桥 |
| shape 推导 | — | ❌ 缺 | 仓内无 infershape 文件；shape 校验在 aclnn CheckShape 承担，GEIR 侧依赖 DynamicRank/UnknownShapeFormat | —（缺省项 1） |
| tiling host | `op_host/chunk_gated_delta_rule_fwd_h_tiling.cpp` + `_tiling.h` + `_tiling_processor.h` | ✅ | Tiling4ChunkGatedDeltaRuleFwdH :79；校验：逻辑维>0 且 `vNumHead % kNumHead == 0`、`vHeadDim ≤ 256`（tiling.cpp:130-144）；**tilingKey = vHeadDim>128 ? 2 : 1**（:152）；processor.h：workspace 布局 6 段（v/vUpdate/kDecay/h/numSeq/numChunks，各按 `aicCoreNum·chunk·dim·4B·2stage` 512B 对齐，:97-121）；`blockDim = aicCoreNum`（:91） | 分核策略、tilingdata、tilingkey |
| 平台配置 | — | ❌ 缺 | 无 config/*.json / simplified_key.ini（tilingkey 仅 1/2 两值，由 vHeadDim 决定） | —（缺省项 2） |
| kernel 入口 | `op_kernel/chunk_gated_delta_rule_fwd_h.cpp` | ✅ | `TILING_KEY_IS(1)→TileShapes128 / (2)→TileShapes256` :118-126；dtype 三级分派（state→gate→useGk/useG）:41-107；**useExp2 恒 false**（:99,103 硬编码） | 模板分支 |
| kernel 实现（arch35 主线） | `op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp`（1071 行） | ✅ | `GDNFwdHTileShapes128{L1=128×128×128}` / `256{L1=128×256×128, L0=128×256×64}` :60-68；**C1: v_work = w@h**（mmad，fp32 累加→GM workspace）:533-654；**C2: h_work = kg^T @ v_update** :656-798；AIV 初始化 h[0]（initial_state 或 0）:814-900；V1/V2 主循环 :939-1028；`useDirectFp32Ub` 直通条件 :248-252；tail 向量兜底 `ComputeTail{V,H}Workspace`（blockTokens<16，纯 FMA 逐行）:353-506 | 片上内存、流水编排 |
| scheduler | `op_kernel/arch35/gemm/block/block_scheduler_gdn_fwd_h.hpp`（425 行） | ✅ | **任务 = (batch, vHead) 对，taskNum = batch·vNumHead**；每核 2 流 ping-pong（`maxTaskCntPerLoop = taskNum > coreNum ? 2 : 1`，:214）；`taskStride = coreNum·2`；流内 chunk 递推串行；offsets 计算 `UpdateTask` :300-330（hSrc/hDst 相邻 chunk 状态块、uvOffset、wkOffset、gOffset）；`NeedProcessStage2 = storeFinalState || !isFinalState` :376-378；CrossCoreFlag 4 对（cube1Done/vec1Done/cube2Done/vec2Done）:125-128；变长解析 `ResolveVarlenSequence`（逐 seq 扫 cu_seqlens）:229-254 | 分核策略、流水编排 |
| epilogue V1（v_new） | `op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp`（573 行） | ✅ | **PrepareG：decay[t] = exp(g_last − g[t])**（标量门，:155-208；useExp2 时先 ×LN2 再 Exp = exp2）；`v_new = u − w@h` 减法 + 行 scale（ApplyRowScale，VF regbase）:358-366；**双写出**：`v_new`（未衰减，CAST_RINT 回 input dtype）+ `v_update`（衰减后，CAST_RINT → zN 供 C2）:369-401；kGated 时把 k 拷入 kDecayWorkspace（KDA 的 kg 由调用方预衰减，注释 :407）；ROW_TILE=16、nv>128 走 wide IO buffer :314-318 | 计算流程、精度策略 |
| epilogue V2（h update） | `op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_update.hpp`（504 行） | ✅ | 标量门 `muls = exp(g_last)`（GetValue 标量 + Exp，:261-283）；kGated：`PrepareKGate` = **exp2(gk_last)**（×LN2+Exp）逐 K 行 scale :350-376；`h_next = h_prev·muls·kg_decay + h_work`（fp32 域 Add :388-390）；fp32 final_state 直通（中间 chunk 也经 finalState GM 走 fp32，:399-421）；末 chunk `hDst=hSrc` 原地（scheduler :312-314） | 计算流程、精度策略 |
| epilogue Reg 基元 | `op_kernel/arch35/epilogue/block/block_epilogue_gdn_fwdh_regbase.hpp`（216 行） | ✅ | MicroAPI RegTensor VF：`PrepareKGateRegbase`（exp2）、`ComputeVNewRegbaseDualIssue`（u−ws 双发射）、`ApplyKGateUpdateRegbaseDualIssue`、`ApplyRowScaleDualIssue`（BRC_B32 广播行 scale）；b16→f32 unpack+zeroing cast trait :20-25 | 流水编排（Reg 级，Triton ❌ 高发区） |
| tiling 数据结构 | `op_kernel/chunk_gated_delta_rule_fwd_h_struct.h` + `_tiling.h` | ✅ | 20 字段：shape 7 + 开关 4 + dtype 3 + workspace offset 6（struct.h:28-52）；kernel 侧 plain struct 与宏生成版字段序一致（struct.h 头注释 :13-21） | tilingdata |
| 文档 | `README.md`（92 行） | ✅ | 公式（:20-27）：`v_new = u − w@h_prev`、`h_next = h_prev·scalar_decay·key_decay + kg^T@v_new`；**gk 固定 log2 空间用 exp2，g 固定自然对数用 exp**（:29-30）；Python API；支持范围：A2/A3/A5、V∈{128,256}、chunk∈{64,128}（:85-91） | 功能分析、规格约束 |
| 测试 | `tests/pta/test_fwd_h.py` + `data_compare_h.py` + `run_gdn_fwd_h.sh` | ✅ | **golden `forward_h_trans_cpu`**（test_fwd_h.py:39-138）：全 fp32 域，chunk 内 cumsum g，公式与 README 一致（:113-115）；g 生成器 `gen_decay_data`（:246-262，base∈[−15,−5]+bias∈[−2,0]，chunk 内 cumsum → g 为 chunk 内累积对数衰减）；比较器 `data_compare`：**diff_thd=1e-4、pct_thd=1e-4（99.99% 达标线）、max_diff cap=0.1**（data_compare_h.py:133,195-205）；入口走 `torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h`（:323） | 测试标杆 |
| 图级语义 | — | ❌ 缺 | 无 op_graph/fusion_pass（非 CANN 仓结构）；in-place 语义证据在 scheduler（`hDst=hSrc` 末 chunk 原地）与 aclnn ViewCopy | —（缺省项 3） |

## 2. 缺省项（进入 P1 前明示）

1. **infershape 缺失**：仓内无独立 infershape；shape 约束以 aclnn `CheckShape`（aclnn_*.cpp:112-169）为准。
2. **平台配置缺失**：无 config json/simplified_key.ini；tilingkey 全集 = {1(V≤128), 2(V=256)}。
3. **图级 pass 缺失**：非 CANN 仓，无 fusion_pass。
4. **⚠️ 依赖出仓（最大缺口）**：kernel 依赖 **Catlass 模板库**（`catlass/gemm/block/block_mmad.hpp`、`block_swizzle.hpp`、`dispatch_policy.hpp`、`epilogue/block/block_epilogue.hpp`、`tla/*`）与 **`kernel_utils/`**（`block_mmad_pingpong_tla{,_multi,_preloadA_l1B}.hpp`、`vector/regbase.hpp`）——均**不在下载快照内**（属 fla_npu 仓其他目录或外部 Catlass）。mmad block 的内部流水（L1/L0 tile 搬运、pingpong 细节）与 regbase VF 的寄存器级实现**不可锚**，P1 对这些只能抽「接口级事实 + 调用参数」，P2 判 ❌ 时需注明「机制细节不可得，按黑盒 Mmad 处理」。如需补齐，可从同仓 `third_party/` 或 Catlass 上游再下。
5. **A2 变体为副线**：`op_kernel/{gemm,epilogue}/` 顶层文件（AtlasA2）只抽差异，不逐行锚。

## 3. P1 抽取要点提示

- 公式链必须与 golden `forward_h_trans_cpu`（test_fwd_h.py:95-130）逐行对齐：**g 是 chunk 内累积对数衰减**（cumsum 语义，gen_decay_data:261 佐证），decay[t]=exp(g_last−g[t]) 作用于 v_new 后再被 kg^T 乘；h_prev 只乘 chunk 末值 exp(g_last)·exp2(gk_last)。
- 三种 gate 组合（useG/useGk/both）在 kernel 分派层只有 3 分支（`useGk&&useG` / `useGk&&!useG` / `!useGk`（必 useG），kernel .cpp:49-60）——注意 `!useGk` 分支模板实参是 `<false, true>`，**不存在 gk-only 之外的 scalarGated=false 单独路径**；scalarGated=false 仅出现在 useGk-only（KDA）场景。
- `v_new`（输出）与 `v_update`（C2 输入）是**两个不同的张量**：前者未衰减、后者衰减（epilogue 双写出），Triton 化时勿合并。
- CV 握手协议：AIC 等 vec2Done 才复用 v_work buffer（C1）/ 等 vec1Done 才跑 C2；AIV 等 cube1Done/cube2Done；`SyncAll` 开场（fwd_h 可作为 megakernel 后续 stage，kernel :509-511 注释）。
