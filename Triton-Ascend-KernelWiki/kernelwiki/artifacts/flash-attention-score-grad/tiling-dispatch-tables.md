# FlashAttentionScoreGrad - Tiling / 核切分 / 模板 dispatch (S4 + S5)

## A. CANN 原始 tiling / 核切分表 (S4)

逻辑计算轴：B、N2（kv head）、G（query/kv head 比 = N1/N2）、S1（query seq）、S2（kv seq）、D（head dim）、Dv（value head dim）。切分顺序：**先核内再核间**（`docs/FAG算子设计介绍.md` L118-126）。

### A.1 arch32（Atlas A2 训练 / ascend910b / ascend910_93）

| 模板 | tiling 类 / kernel 文件 | 触发条件（shape） | 核内切分轴 | 核间切分轴 | CV 基本块 | tiling key |
|------|------|------|------|------|------|------|
| S1S2（通用泛化） | `TilingS1s2Bn2gs1s2` / `..._s1s2_bn2gs1s2.h` | 支持所有 shape，其他模板不命中则走此（`FAG算子设计介绍.md` L268-279） | S1, S2 | B, N2, G, S1 | C 128×1024 / V 8×1024（nRatio=8，1:16） | `0x1434` when IsSameAB=0; UB0=4,UB1=3,Block=4 |
| SameAB（_sab，Cube 为主核） | `TilingS1s2Bn2gs1s2SameAB` / `..._sab.h` | 确定性场景（非FP32 或 S1≥1024 或 S2≥1024）；TND 平均 S1≥1024 且 S2≥1024；普通 S1≥1024 或 S2≥1024（`FAG算子设计介绍.md` L162-169） | S1, S2 | S2 | 同上；NZ 输出可复用左矩阵 L0C 累加 | `0x1434` (UB0=4,UB1=3,Block=4,IsSameAB=1) |
| BN2 | `TilingS1s2Bn2` / `..._s1s2_bn2.h` | (S1<1024)&&(S2<1024)&&(B*N2*2>CoreNum ‖ G==1)（`FAG算子设计介绍.md` L283-291） | G, S1, S2 | B, N2 | 同上 | `0x134` (UB0=4,UB1=3,Block=1) |
| NGS1S2_BN（核内切 N2.i） | `TilingUngs1s2Bbn` / `..._ngs1s2_bn.h` | G*S1*S2≤32KB && G==1 && S2<1536（`FAG算子设计介绍.md` L295-305） | N2.i, G, S1, S2 | B, N2.o | 同上 | `0x199` (UB0=9,UB1=9,Block=1) |
| B（核内 B.i） | `TilingUbngs1s2Bb` / `..._bngs1s2_b.h` | N2*G*S1*S2≤64*128（`FAG算子设计介绍.md` L309-319） | B.i, N2, G, S1, S2 | B | 同上 | `0x099` (UB0=9,UB1=9,Block=0) |
| Basic / MLA | `TilingS1s2Bn2gs1s2Basic` / `..._basic.h` | 小 shape 兜底/MLA 场景（低阶 API，Cube 为主核） | - | - | - | `0x999` (UB0=9,UB1=9,Block=9) |
| Basic_det（确定性） | `TilingS1s2Bn2gs1s2BasicDet` / `..._basic_det.h` | 开启确定性计算（`FAG算子设计介绍.md` L357-366） | BN2GS1S2 特定分核避免同址累加 | 同左 | - | `0x1999` (UB0=9,UB1=9,Block=9,IsDeterministic=1) |
| Empty tensor | `RunEmptyTiling` / `..._empty_tensor.h` | dq/dk/dv 任一为空（`tiling.cpp` L237-256,383-392） | - | 按 aivNum 切 dq/dk/dv/dpse 清零 | - | `FAG_EMPTY_TILING_KEY=0` |

> 注：设计文档的 5 模板表（B/N2/SameAB/S1S2/TND，`FAG算子设计介绍.md` L148-184）是 Vector 视角的简化；kernel 实际 dispatch 树（`op_kernel/flash_attention_score_grad.cpp` L508-901）含上述 8 分支。FP32 dtype 仅支持 S1S2/BN2/NGS1S2_BN/B（不支持 SameAB/Basic/Det，见 L837-901）。

### A.2 arch35（Ascend 950PR/950DT，regbase）

| 模板 | 触发条件 | 核间切分轴 | CV 基本块 | 文件 |
|------|------|------|------|------|
| BN2 | S1≤128 and S2≤128 and G==1，非 FP32（`FAG算子设计介绍.md` L343-344；`...regbase.cpp` L94 `BN2_MAX_S=128`） | BN2 | C 128×128 / V 64×128（C:V=1:2） | `kernel_base.h` |
| BN2S2 | 非 FP32，TND，B*N2*S2>cubeCoreNum and G==1（`FAG算子设计介绍.md` L354-355；`...regbase.cpp` L95 `BN2S2_MAX_S=1024`） | BN2S2 | 同上 | `kernel_base.h` |
| BN2GS1S2（通用泛化） | 上述都不满足（支持所有 shape） | BN2GS1S2 | 同上 | `kernel_base.h` |
| 确定性计算 | 开启确定性计算 | BN2GS1S2（特定分核） | 同上 | `kernel_deter.h` |

arch35 仅 2 个 kernel 文件（非确定性 `kernel.h`/`kernel_base.h`、确定性 `kernel_deter.h`），BN2/BN2S2/BN2GS1S2 通过模板参数隔离（`FAG算子设计介绍.md` L322）。CV 通路：AIC 直出 UB（`fixp_l0c_to_ub`），Vector 结果经 `copy_ub_to_l1` 回 L1 供 dq/dk/dv matmul（`FAG算子设计介绍.md` L64-68）。

### A.3 UB / Workspace 分配

- arch32 `PreParams`/`PostParams`（`tiling_common.h` L272-377）：`maskPreBlockTotal`、`qPreBlockFactor/Tail`、`kvPreBlockFactor/Tail`、`castBufferLen`、`outputBufferLen`、`inputBufferLen`、`singleUBProcessNum`、`dropoutIsDivisibleBy8`；Post 段含 `dqWorkSpaceOffset`/`dkWorkSpaceOffset`/`dvWorkSpaceOffset`/`dqRopeWorkSpaceOffset`/`dkRopeWorkSpaceOffset`（L362-366）。
- arch35 `REGBASE_POST_BASE=128*128`（`...regbase.cpp` L106）、`RESERVED_WORKSPACE_SIZE=64KB`（L26）、`BN2S2_WRITE_UB_D=128`（L100）。
- 空 tensor：workspace 预留 **100MB**（`tiling.cpp` L51 `WORKSPACE_SIZE=100*1024*1024`）。
- 正常场景 workspace 分区存放 mm1 重算结果与 dq/dk/dv 中间结果（Post 段 offset）。
- V 侧 double buffer（ping-pong），`FAG算子设计介绍.md` L78-91。

### A.4 核间 / BlockDim

- `KERNEL_TYPE_MIX_AIC_1_2`（1 Cube : 2 Vector，`op_kernel/flash_attention_score_grad.cpp` L501）。
- `CalculateTschBlockDim(sliceNum, aicNum, aivNum)`：aiv/aic 比值 ration，`blockDim=(sliceNum+ration-1)/ration`（`tiling.cpp` L54-62）。
- 多核用满、负载均匀、CV 算力匹配；S2 不切到多核（softmax 在线更新需连续 S2，`FAG算子设计介绍.md` L99 适用于前向，反向同样保留 S2 为循环轴）。

## B. Triton-Ascend 抽象映射 (S4)

| CANN 概念 | Triton-Ascend 抽象 |
|---|---|
| `blockDim` / core id / AIC:AIV 1:2 | `grid` 形状 + `tl.program_id(axis)`；AIV-only 清零路径->1D grid |
| 核间切分轴（B/N2/G/S1） | `grid` 维度；`program_id` 映射到 (b, n2, g, s1_block) |
| `S1.i`/`S2.i`/`N2.i` CV 基本块 | `tl.constexpr` `BLOCK_M`(=S1.i)/`BLOCK_N`(=S2.i)/`BLOCK_K`(=D) |
| 核内尾块（blockNum*BLOCK<total） | `mask = offsets < n` masked load/store |
| tiling key -> 模板选择 | launcher 按 shape/dtype/mask 选 `BLOCK_*` 与 `@triton.jit` kernel；`if constexpr` 分支 |
| 空 tensor 100MB workspace + 清零 | 启动器短路：`tl.store` 零张量，无 matmul |
| 重算 p 的 mm1 workspace | drop（Triton 管理 temp）/ on-chip `tl` tensor |
| FP32 累加（L0C/UB FP32 中间） | `tl.dot(a,b).to(tl.float32)` 累加器，输出 cast 回 dtype |
| MTE/BufferID/queue/event、`fixp_l0c_to_ub`/`copy_ub_to_l1` | 抽象为 tile 内复用 + fused load/store；不移植 |
| `ascendcPlatform` 运行时查询 | launcher 按 shape 选 `BLOCK_*`；skeleton 不查询 |
| V 侧 double buffer / Cube preload | 编译器排流；skeleton 只表达循环与依赖 |

## C. 模板与 dispatch 计数 (S5)

- **arch32 kernel 模板文件**：8 类（S1S2、SameAB、BN2、NGS1S2_BN、B、Basic、Basic_det、Empty），分别对应 `op_kernel/arch32/flash_attention_score_grad_{s1s2_bn2gs1s2, _s1s2_bn2gs1s2_sab, _s1s2_bn2, _ngs1s2_bn, _bngs1s2_b, _s1s2_bn2gs1s2_basic, _s1s2_bn2gs1s2_basic_det, _empty_tensor}.h`。命名后缀 `_sab`=SameAB(Cube 主核)，`_basic`/`_basic_det`=低阶 API/确定性。
- **arch35 kernel 模板文件**：2 个（`kernel.h`/`kernel_base.h` 非确定性、`kernel_deter.h` 确定性），另 `kernel_quant.h` 处理 FP8；BN2/BN2S2/BN2GS1S2 用模板参数隔离。
- **dispatch 维度**（20 个 kernel 模板参数，`op_kernel/flash_attention_score_grad.cpp` L488-490）：`UB0,UB1,Block,IsSameAB`（模板路由）、`DataType`（FP16/BF16/FLOAT，FP8 走 arch35 quant）、`Layout`(0=BSNGD族/1=SBNGD/2=BNGS1S2/3=TND)、`Sparse`、`MatmulCfg`(NORMAL/EXCEED)、`Mm12IsNZOut,Mm345IsNZOut`(NZ 输出)、`HasDropOut,HasPse,HasAttenMask`、`EnableL1Reuse`、`TNDS1Pingpong`、`S1TemplateType,S2TemplateType,DTemplateType`(对齐特化 Aligned16/32/48/64/128/192、NotAligned30/60/72/88)、`IsDeterministic`、`HasRope`。
- **dispatch 触发仍可用于 Triton-Ascend 的**：dtype / layout / has_mask / has_pse / has_dropout / has_rope / deterministic / sparse_mode -> 选 constexpr 分支与 `BLOCK_*`。
- **CANN 编译/注册机制（不移植）**：tiling key bit 编码、`*_binary.json`、`simplified_key.ini`、`REGISTER_TILING_FOR_TILINGKEY`、`KERNEL_TYPE_MIX_AIC_1_2` 核型绑定。

## D. aclnn 接口版本 dispatch（来源：`docs/aclnnFlashAttentionScoreGrad*.md`、`op_api/aclnn_flash_attention_score_grad.cpp`）

| 接口 | 架构支持 | 相对前一版新增 |
|------|---------|----------------|
| `aclnnFlashAttentionScoreGrad` (V1) | A2 训练 √ / 950PR × | 基线（pseType=1 语义） |
| `aclnnFlashAttentionScoreGradV2` | A2 训练 √ / 950PR × | 新增 `pseType` 参数 |
| `aclnnFlashAttentionScoreGradV3` | A2 训练 √ / 950PR ×（950PR 暂不支持 sink） | 新增 `sinkInOptional` 输入与 `dsinkOut` 输出 |
| `aclnnFlashAttentionScoreGradV4` | 950PR √ / A2 × | 合并 V2 + UnpaddingV2；多等/不等长序列（varlen）；Dropout 调整（keepProb<1 且无外部 mask 时自动生成） |
| `aclnnFlashAttentionUnpaddingScoreGrad` V1-V5 | 变长/TND | varlen（actual_seq_qlen/kvlen、q/kv_start_idx）族，V5 含完整 RoPE |

> operator 级产品支持（`README.md` L5-11）：950PR/950DT √、Atlas A3 训练 √、Atlas A2 训练 √；A2/A3 推理 ×。FP8（FLOAT8_E5M2/E4M3FN/HIFLOAT8）仅 arch35(950) 支持，且不可配 pse/dropMask/attenMask/keepProb（`README.md` L179-181）。
