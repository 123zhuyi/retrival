# prompt_flash_attention — tiling 与 dispatch 分析

来源：`attention/prompt_flash_attention/`（commit `d28d8f9...`）。所有结论标路径+行号。tiling 参数具体数值属 `needs_verification`（依赖运行时 shape/平台）。

## A. 架构分发（host -> kernel）

`def.cpp` L104-127 注册 AICore config：`ascend910b` 与 `ascend910_93` 用 opFile `prompt_flash_attention`（arch32 内核），`ascend950` 用 opFile `prompt_flash_attention_apt`（arch35 内核），`ascend310p`（Atlas 推理加速卡，仅 FP16）单独 config。`tiling.cpp` 内以 `curShortSocName == SocVersion::ASCEND310P` / `ASCEND910B` 做大量分支（L429/433/1469/3920 等）。

| 架构 | SocVersion | tiling 路径 | kernel 入口 | 说明 |
|---|---|---|---|---|
| Atlas A2 训练/推理 | ASCEND910B | `tiling.cpp`（arch32 tiling） | `prompt_flash_attention_arch32.h` | 多模板（S1S2/BNSD tilling/MLA/high-precision/empty） |
| Atlas A3 训练/推理 | ASCEND910_93 | 复用 arch32 | arch32 | 同 A2 kernel |
| Ascend 950PR/950DT | ASCEND950 (DAV_3510/arch35) | `tiling_v2.cpp`（arch35 tiling） | `arch35/prompt_flash_attention_entry_regbase.h` | 服务化结构（mm1/vector1/mm2/vector2）+ 模板参数化 |
| Atlas 推理加速卡 | ASCEND310P | `tiling.cpp` 310P 分支 | arch32 310TILING | 仅 FP16，`defaultSysWorkspaceSize = libApiWorkSpace`（register.cpp L44-49） |

## B. arch32 tilingkey（十进制组装，CANN 编译期模板注册）

`prompt_flash_attention_tilingkey.h` 定义 ~130 个 tilingkey 宏（如 `QFP16_KVFP16_OUTFP16_BNSD_HIGHPERFORMANCE_HIGHLEVELAPI_MDL_NOTAIL_NEWTILING = 1000000000000000011`）。组装维度（设计介绍 L61-92）：

| 维度 | 取值（示例） | 魔数/位域 |
|---|---|---|
| 场景标识 | A2/A3=1012 或 10；推理加速卡=12288 | 十进制前缀 |
| Q dtype | FP16/BF16/INT8/FP8/FP4E1M2 | +600(FP16高精度/MSD), +100(BF16), +800000000000(INT8非MSD) |
| KV dtype | FP16/INT8/FP8 | 嵌入 Q 维 |
| OUT dtype | FP16/BF16/INT8 | +10000(BF16), +20000(INT8) |
| layout | BSH/BSND/BNSD/TND/NTD_TND/PA | +100000(BSH/BSND), +5(BNSD, 非FP16尾块场景) |
| 精度模式 | HIGHPERFORMANCE/HIGHPRECISION | +600(高精度) |
| API 级别 | HIGHLEVELAPI/BASICAPI | 2× 前缀 |
| 功能叠加 | MLA / PA(PagedAttention) / PREFIX(共享前缀) / MSD伪量化 / MDL / CUBEVECTORDIFF / NORMAL | +10000000(PA), +100000000(PREFIX), +400200000000(MSD) |
| 尾块 | TAIL/NOTAIL | 后缀 |
| tiling 变体 | NEWTILING/OLDTILING/310TILING/EMPTYKVTILINGKEY | EMPTYKV=20（tiling.cpp L73） |

> 该十进制 tilingkey 是 CANN 编译期二进制/模板注册机制，**不迁移**到 Triton；Triton 用 `if constexpr`/launcher dispatch 或独立 `@triton.jit` kernel 表达等价分支。

## C. arch35 模板参数化（template_tiling_key_enum.h）

arch35 用结构化枚举做 `if constexpr` 模板分发（比 arch32 十进制 key 更清晰）：

| 参数 | 枚举 | 来源 |
|---|---|---|
| Layout | BSH/BNSD/TND/NTD（PFA）；FA 另有 SBH/NTD_TND | L51-52 |
| S1 模板 | 16/32/64/128/256/NotAligned | L69-76 |
| S2 模板 | 16/32/64/128/256/512/1024/NotAligned | L78-87 |
| D 模板 | 16/32/48/64/80/96/128/160/192/256/512/**576**/NotAligned | L53-67（576=MLA nope512+rope64） |
| DV 模板 | 同 D | L121-126 |
| ImplMode | AA_HIGH_PRECISION(0)/AA_HIGH_PERFORMANCE(1)/AA_INVALID_LINE_HIGH_PRECISION(2) | L89-93 |
| PseMode | OUTER_MUL_ADD(0)/OUTER_ADD_MUL(1)/INNER_MUL_ADD(2)/INNER_MUL_ADD_SQRT(3)/NONE(9) | L189-194 |
| QuantMode | NoQuant(31)/FullQuant(30)/PER_CHANNEL(0)/PER_TOKEN(1)/K_PER_CHANNEL_V_PER_TOKEN(2)/PER_TOKEN_HEAD(3)/PER_TOKEN_PAGE_ATTENTION(4)/... | L197-206 |
| Mask | DISABLE(0)/ENABLE_NO_BAND(1)/ENABLE_BAND(2) | L209-211 |
| MatMulType | MM_PFA(0)/MM_PA(1, 分页)/MM_IFA_MLA(2)/MM_IFA_MLA_PA(3)/MM_PA_D512(4)/MM_DN(5) | L214-219 |
| SplitCoreMode | SPLIT_NBS_VECTOR(0)/SPLIT_NBS_CUBE(1)/SPLIT_ONEN_VECTOR(2)/SPLIT_ONEN_CUBE(3)/BALANCE_VECTOR(4)/BALANCE_CUBE(5) | L222-227 |

`ConfigValue[27]`（L129-157）预定义 27 组 (S1,S2,D,DV) constexpr 组合，如 config 7 = (128,128,512,512)、config 9 = (64,128,576,512)（MLA）。

## D. 核间切分（core partition）

| 项 | 结论 | 来源 |
|---|---|---|
| 逻辑轴 | B、N2(kv head)、G(q/kv head 比)、S1(q seq)、S2(kv seq) | 设计介绍 L49 |
| 切分顺序 | 先核内再核间 | 设计介绍 L49 |
| splitCoreMode 选择 | `enableNBSByCubeSeqMode = actualSeqLength >= seq3K(3072) && b*n*sOuterLoopByCube >= coreNum` → SPLIT_ONEN_CUBE；否则 SPLIT_NBS_CUBE | tiling.cpp L902-932 |
| 负载均衡 | `singleCoreSize = (totalSize + curCoreNum - 1) / curCoreNum`（向上取整） | tiling.cpp L713 |
| 多核用满/均匀/CV 算力匹配 | 三原则 | 设计介绍 L43-47 |
| S2 是否切到多核 | 标准路径 S2 不切到多核（FlashSoftmax 需连续 S2 在线更新）；PA/FlashDecoding 场景 `needs_verification` | 设计介绍 L13；对比 FA |

## E. 核内 tiling 基本块

| 单元 | 基本块 | 来源 |
|---|---|---|
| Cube (AIC) C-Tiling | 128×128（FP16 入，FP32 出） | tiling.cpp L150-151；设计介绍 L59 |
| Vector (AIV) V-Tiling | 8×1024（FP32，32KB 上限） | 设计介绍 L59 |
| CV 配比 | nRatio=8 → Cube 一轮 128×1024，Vector 分 16 次 8×1024 | 设计介绍 L59 |
| S1(sOuter) factor | 默认 128（CVDIFF_SOUTER_FACTOR_DEFAULT） | tiling.cpp L103 |
| S2(sInner) factor | 默认 1024（CVDIFF_SINNER_FACTOR_DEFAULT）；小 KV 阈值 1024 | tiling.cpp L104/106 |
| mm1Res UB | 16384B = 128×128（CVDIFF_MM1RES_UB_SIZE） | tiling.cpp L102 |
| BLOCK_SIZE_BASE/MAX | 128 / 1024 | tiling.cpp L93-94 |
| S1 负载候选 COF | {256,384,512,640,768,896,960,1024}（arch32 按 coreNum 选） | tiling_const.h L63 |

## F. MLA / 空 / PA 触发

| 分支 | 触发条件 | 来源 |
|---|---|---|
| MLA 模式 | `headSize != headSizeV` 或 `headSize > MLA_THRESHOLD(256)` → CVDIFF_MLA | tiling.cpp L159/2848 |
| 空 KV | emptyTensor 路径，tilingkey=20 | tiling.cpp L73/350 |
| PagedAttention | blockTable 非空时启用；`blockNumValid = Σ ceil(actualSeqKV / blockSize)` | tiling.cpp L1959/1987-1988 |
| 共享前缀 | keySharedPrefix/valueSharedPrefix/actualSharedPrefixLen 输入（kernel 入口 L34） | prompt_flash_attention.cpp L34 |

## G. Workspace

`workspaceSize = sysWorkspaceSize + coreNum × softmaxDtypeSize × (maxSpmSize + mm1ResSize×2 + mm2ResSize×2)`（+ antiquant + MDS 扩展）（tiling.cpp L465-490）。`×2` = double buffer。910B sysWorkspace=16MB（L447/502），WORK_SPACE_RESERVE=16MB（L154）。

## H. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| `blockNum`/core id | `grid` + `tl.program_id(axis)` | B/N2/G/S1 轴映射到 program 维 |
| splitCoreMode (NBS/ONEN × VECTOR/CUBE) | grid 形状 + program_id 轴选择 | AIV-only 可 1D grid；AIC/AIV 分离 → Triton 单 program family |
| S1/S2 基本块 (128/1024) | `tl.constexpr` (`BLOCK_M=128`, `BLOCK_N=1024` 等) | 静态 meta，值 source-reported |
| tilingkey 分支 | `if constexpr` / 独立 kernel / launcher dispatch | 十进制 key 本身不迁移 |
| MLA (D=576, nope+rope) | D 维拼接的 constexpr + 两次 dot | 抽象为 Q/K 沿 D 拼接 |
| 空 tensor | early-return / mask 全 False | 边界 |
| workspace (mm1Res/mm2Res ×2) | 中间 `tl` tensor 或丢弃 | Triton 管理临时 |
| CV nRatio=8 | Triton 层不显式（编译器调度） | CANN 专有，不迁移 |
| tail block | `mask = offsets < n` on load/store | 边界 |
| TilingPrepare 平台查询 | launcher 按 shape 选 BLOCK_* | 运行时查询不迁移 |
