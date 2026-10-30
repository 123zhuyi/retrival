# sparse_flash_attention - tiling 与 dispatch 分析

来源：`attention/sparse_flash_attention/`（commit `d28d8f9...`）。深度 tiling/分核分析见 [[technique-sparse-attention-tiling]]，本文仅补 dispatch 结构与 Triton 映射。

## A. 架构与模板分发

`def.cpp` L96-97 仅注册 `ascend910b` + `ascend910_93`（A2/A3 推理）。无 arch32/arch35 子目录，单架构实现。

模板参数（`sparse_flash_attention_template_tiling_key.h` L31-37）：

| 参数 | 取值 | 说明 |
|---|---|---|
| FLASH_DECODE | 0/1 | FlashDecoding 核间 S2 切分开关 |
| LAYOUT_T | BSND(0)/TND(1) | query layout |
| KV_LAYOUT_T | BSND(0)/TND(1)/PA_BSND(2) | kv layout（PA_BSND=PagedAttention） |
| TEMPLATE_MODE | C_TEMPLATE(0)/V_TEMPLATE(1) | Cube 主 / Vector 主（含 MergeKv） |

`ASCENDC_TPL_SEL`（L41-69）定义 4 组合法组合（FLASH_DECODE=0 × {BSND,TND} × {C,V}_TEMPLATE，KV 含 PA_BSND）。dtype 分发在 kernel 入口 `if constexpr`（FP16/BF16，sparse_flash_attention.cpp L46-53）。

C/V 模板选择（host 侧 `InitParams`）：`sparseBlockSize ≤ 4 && s2Size ≠ 0` -> V_TEMPLATE，否则 C_TEMPLATE（见 [[technique-sparse-attention-tiling]] §3.1）。

任务类型：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`（kernel 入口 L41）-> AIC:AIV = 1:2。

## B. 核内/核间切分（概要，详见 [[technique-sparse-attention-tiling]]）

- 逻辑轴 B/N2/G/S1/S2；`ProcessBalance` 按 B/N2/G/S1 四轴负载均衡（kernel_mla.h L785-849）。
- 基本块：`mBaseSize`=128（S1），`s2BaseSize` 动态计算（S2），`N_BUFFER_M_BASIC_SIZE`=256，`PRELOAD_NUM`=2。
- MLA：D(nope)=512 + D(rope)=64 = 576，headDimAlign=Align(qkHeadDim,32B)。
- FlashDecoding：`splitKVFlag` 为 true 时 S2 参与核间切分（与标准 FA 的“S2 不切到多核”不同，因稀疏 block 的 softmax 可独立计算后合并）。
- UB 预算：`mmResUb + bmm2ResUb + softmax 临时 ≤ ubSize`（`CalcUbBmm`，tiling.cpp）。

## C. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| `blockNum`/core id（B/N2/G/S1 切分） | `grid` + `tl.program_id` | 四轴映射 program 维 |
| FlashDecoding S2 切分 | program_id 额外轴 + 后处理 reduce | 稀疏 block softmax 可独立合并 |
| mBaseSize=128 / s2BaseSize | `tl.constexpr`（BLOCK_M/BLOCK_N，值 source-reported） | 静态 meta |
| TEMPLATE_MODE (C/V) | `if constexpr` / 独立 kernel | C/V 模板分支 |
| sparse_indices 驱动 S2 | `tl.load(sparse_indices)` -> 间接 `tl.load` KV | 离散 gather -> 连续 chunk |
| MergeKv（V_TEMPLATE） | gather 到连续 tl tensor 再 dot | 抽象为 tile 级 gather |
| MLA D=576(nope+rope) | D 维拼接 constexpr + 分段 dot | Q/K 沿 D 拼 nope+rope |
| CrossCoreSetFlag/WaitFlag（5 event） | 不迁移（CANN 手动同步） | 抽象为 tile 依赖 |
| PreloadPipeline 3-slot | 不迁移（CANN 流水机制） | 抽象为 tile 复用/fused load-store |
| ND↔NZ / DataCopyPad | 不迁移 | Triton layout（编译器） |
