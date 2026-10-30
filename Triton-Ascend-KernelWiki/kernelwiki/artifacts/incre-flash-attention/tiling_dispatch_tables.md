# incre_flash_attention - tiling 与 dispatch 分析

来源：`attention/incre_flash_attention/`（commit `d28d8f9…`）。所有结论标路径+行号。tiling 参数具体数值属 `needs_verification`（依赖运行时 shape/平台）。IFA 为增量解码（query S1=1），核间招牌机制为 FlashDecode（split-KV + lse 规约）。

## A. 架构分发（host -> kernel）

`def.cpp` L294-408 注册 AICore config：

| 架构 | SocVersion / config | opFile（kernel 入口） | tiling 路径 | 说明 |
|---|---|---|---|---|
| Atlas A2 训练/推理 | `ascend910b` L303 | `incre_flash_attention`（arch32） | `tiling.cpp` | arch32 主体，三模板分发 |
| Atlas A3 训练 | `ascend910_93` L304 | `incre_flash_attention`（arch32） | `tiling.cpp` | 复用 arch32 |
| Ascend 950PR/950DT | `ascend950` L306-316 | `incre_flash_attention_apt`（arch35） | `tiling_v2.cpp` | arch35 regbase，服务化（mm1/vec1/mm2/vec2）；`jitCompile static_false,dynamic_false` |
| Atlas 推理加速卡（310P） | `ascend310p` L318-408 | `incre_flash_attention` | `tiling.cpp` 310P 分支 | 仅 FP16（部分 INT8 KV），经 arch32 入口转 `arch20/*` 解码器（关系 `needs_verification`） |

> V4 doc L9 标注 950PR/950DT ×（V4 的 KV 左 padding 特性），但 `def.cpp` L316 注册了 950 config；V4 接口在 950 的支持范围 `needs_verification`。

## B. arch32 TilingKey（十进制 16 位组装，CANN 编译期模板注册）

`incre_flash_attention_tilingkey.h` L18-454 定义 tilingkey 宏；组装规则见设计介绍 L189-244（`GenTilingKey` L3409-3470）：

| 十进制位 | 变量 | 取值 | 来源 |
|---|---|---|---|
| 0 | layoutVal | Q shape：0:BNSD / 1:BSH,BSND / 2:TND | 设计介绍 L230 |
| 1 | inputQVal | query：0:FP16 / 2:BF16 / 3:INT8 | L231 |
| 2 | inputKvVal | KV：0:FP16 / 2:BF16 / 3:INT8 / 4:INT4 | L232 |
| 3 | outputVal | output：0:FP16 / 2:BF16 / 3:INT8 | L233 |
| 4 | originVal | 同 inputQVal | L234 |
| 5[bit0] | splitKvVal | FlashDecode：1:enable / 0:disable | L235 |
| 5[bit1] | paVal | PageAttention：1:enable / 0:disable | L236 |
| 5[bit2] | antiquantModeVal | PerToken 伪量化：1:enable / 0:disable | L237 |
| 6 | antiquantMode_ | 0:无效 / 2:K-perChannel-V-perToken | L238 |
| 7 | kvLayoutVal | KV shape（仅 DD/MLA 模板有效）：0:BNSD / 1:BSH,BSND / 2:NZ | L239 |
| 8 | amlaMode | 废弃，恒 0 | L240 |
| 9 | balanceMode | 负载均衡（仅 MLA 全量化模板可使能） | L241 |
| 10–14 | 预留 | 0 | L242 |
| 15 | perfMode_ | **模板编号**：0:C1_V2(CV 1:2) / 1:全V / 2:C1_V1(CV 1:1) / 3:matmul基础API / 5:MLA全量化 / 6:伪量化MSD DD | L243 |
| 16 | modeVal | 1:IFA Base / 2:IFA SysPrefix | L244 |

> 该十进制 tilingkey 是 CANN 编译期二进制/模板注册机制，**不迁移**到 Triton；Triton 用 `if constexpr`/launcher dispatch 或独立 `@triton.jit` kernel 表达等价分支。IFA 实际只用 perfMode_ ∈ {0,1,3}（C+V / 全V / matmul基础API）；{5,6}（MLA全量化 / DD）设计介绍 L54/L56 标注仅 FIA 调用。

## C. arch35 模板参数化（`template_tiling_key.h`）

arch35 用结构化模板参数做 `if constexpr` 分发（`incre_flash_attention_template_tiling_key.h` L35-119），比 arch32 十进制 key 更清晰，参数维度与 PFA arch35 同源（layout / S1 / S2 / D / ImplMode / PseMode / QuantMode / Mask / MatMulType / SplitCoreMode）。IFA arch35 因 S1=1，S1 模板维度退化为常量。**具体枚举值与 ConfigValue 组合 `needs_verification`**（未在本分析中逐一提取，以 `template_tiling_key.h` L35-119 为准）。

## D. 核间切分（core partition）

来源：设计介绍 L71-75。

| 项 | 结论 | 来源 |
|---|---|---|
| 逻辑轴 | B、N2（kv head）、G（q/kv head 比）、S1（q seq，**固定 1 不切**）、S2（kv seq） | 设计介绍 L71 |
| G 轴 | 仅在 Vector 计算时切块（UB 驱动） | L71；L160-173 |
| 核间策略 | 先按 BN2 分核；当 `BN2 < 0.4 × 总核数` 时再对 S2 外切 SplitKv 份，总块数 = `BN2 × SplitKv` | L73 |
| FlashDecode | S2 外切后各核独立做 Attention，最后对子块结果 reduce 规约（lse 加权） | L73/119-138 |
| 核内 | 单 core 缓存有限，对 S2（或 KV 子块的 S2）切分，即 FlashAttention 循环 | L75 |
| 分核负载均衡 | 按块负载（每块 S 实际长度）对连续块组合重排，使核间负载差最小；输出 blockid 数组 | L180-188 |
| 三原则 | 核数用满 / 各核负载均匀 / AIC-AIV 算力匹配 | L65-69 |

## E. 核内 tiling 基本块

IFA query S1=1，Cube Bmm1（Q·K^T）的 M 维退化为 1；Vector 侧沿 S2 分块处理。具体基本块数值 `needs_verification`（依赖 `tiling.cpp`/`tiling_base.h` 常量与运行时 shape/平台）：

| 单元 | 推断 | 来源/状态 |
|---|---|---|
| Cube Bmm1 (Q·K^T) | M=1（S1=1）× BLOCK_N（S2 分块），FP32 累加 | 推断（S1=1）；数值 needs_verification |
| Cube Bmm2 (P·V) | BLOCK_N × DV，FP32 累加 | 推断；数值 needs_verification |
| Vector V-Tiling | 沿 S2 分块，FP32 中间（最大 ~32KB） | 设计介绍 L59（PFA 同源）；IFA 数值 needs_verification |
| GQA G 切分 | `g = target_ub_size / column_size`，`g > G` 时取 G | 设计介绍 L166-172 |

## F. 模板选择 + 特性触发

| 分支 | 触发条件 | 来源 |
|---|---|---|
| C+V 模板（perfMode 0） | 基础模板，绝大多数场景；Cube 做 matmul（高阶 API） | 设计介绍 L40 |
| 全 Vector 模板（perfMode 1） | 推理加速卡全用；A2 场景：非 PA、非 GQA、Q/KV/Out 全 FP16 | 设计介绍 L41-45 |
| matmul 基础 API 模板（perfMode 3） | `EnableCubeViewMM` 判定；CUBE 视角 + N-Buffer 流水 | 设计介绍 L46-52 |
| FlashDecode（splitKv） | `BN2 < 0.4 × 总核数`（设计介绍）；`tiling.cpp` `IsFlashDecode` L988/L2948 | L73 |
| PagedAttention | `block_table` 非空（`tiling.cpp` L3847）；blockSize>0，16/32 对齐 | 设计介绍 L156-158；V4 doc L461-469 |
| MLA | `tiling.cpp` L297（`perfMode_ == CUBE_VIEW_MM_MLA`）；IFA 实际是否走 MLA `needs_verification` | L297；设计介绍 L56 |
| AntiQuant-MSD | antiquant_scale/offset 非空 + INT8 KV；A 变换适配 B | 设计介绍 L140-155 |
| GQA | `numHeads > numKeyValueHeads`（G>1），UB 不足时 G 轴切分 | 设计介绍 L160-173 |
| KV 左 padding（V4） | `kv_padding_size` 非空；起点=`Smax - kvPadding - actualLen`，终点=`Smax - kvPadding` | V4 doc L255-263 |
| 空 KV | early-return / 输出全零 | 推断（PFA 同源）；needs_verification |

## G. Workspace

`workspaceSize` 由 `CalcWorkSpace`（`tiling.cpp` L3597-3625）计算，含 sysWorkspace + coreNum × (softmax lse/max/sum + mm1Res×2 + mm2Res×2)（`×2` = double buffer）+ PA 场景 `mmPACallBackDataSize×2`（`tiling_v2.cpp` L3620-3622）+ antiquant/MSD 扩展。**精确字节预算 `needs_verification`**（依赖平台/shape）。

## H. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| `blockNum`/core id 沿 BN2 | `grid` + `tl.program_id(0)` | BN2 维映射到 program 维 |
| FlashDecode split-Kv | `tl.program_id(1)` 沿 S2-split 维 + 规约 kernel | 两段：split 计算 + lse 加权 reduce |
| S1=1 | `BLOCK_M=1` constexpr | 增量解码单 token |
| S2 基本块 | `BLOCK_N: tl.constexpr`（值 source-reported） | 核内 FlashAttention 循环步长 |
| TilingKey 分支 | `if constexpr` / 独立 kernel / launcher dispatch | 十进制 key 本身不迁移 |
| 三模板（C+V/全V/preload） | 3 个 `@triton.jit` kernel 或同 kernel 内 constexpr 分支 | 按 dtype/layout/shape 选 |
| PagedAttention blockTable | `tl.load(block_table)` 查物理偏移 -> 间接 `tl.load` KV | block_size 对齐 |
| AntiQuant-MSD | Q 展开 int8 -> `tl.dot(int8,int8)->int32` -> reduce | A 变换适配 B |
| GQA G 切分 | G 维循环，sub_block 由 constexpr 容量决定 | UB 驱动 |
| KV 左 padding | KV offset = `Smax - kvPadding - actualLen`，越界 mask=0 | 边界 |
| tail block / 变长 actualSeqLen | `mask = offs < actual_len` on load/store | 边界 |
| workspace（mm1Res/mm2Res ×2） | 中间 `tl` tensor 或丢弃 | Triton 管理临时 |
| CV N-Buffer 流水 / Cube 双发 | Triton 层不显式（编译器调度） | CANN 专有，不迁移 |
| MTE/BufferID/queue/event | 不迁移 | 抽象为 tile 复用/fused load-store |
