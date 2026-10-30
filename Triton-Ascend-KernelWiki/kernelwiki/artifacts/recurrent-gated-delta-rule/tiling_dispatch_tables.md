# recurrent_gated_delta_rule - tiling 与 dispatch 分析

## A. 架构分发

`def.cpp` L91-92：仅 `ascend910b` + `ascend910_93`（A2/A3）。READEME L7-12：不支持 950/310P/训练系列。仅 BF16。

## B. 任务类型

`KERNEL_TYPE_AIV_ONLY`（kernel 入口 L31）：**纯 AIV 执行，无 Cube 参与**。这是与 flash attention 系列的最本质区别（RGDR 是线性注意力 SSM 风格，非 QK matmul 主导）。

## C. 分核策略

`tiling_data.h` L23-38：
- `vectorCoreNum`：AIV 核数。
- `b`：batch 大小（`B_ = tilingData->b`，kernel.h L54）。
- `t`：总 token 数（`T_ = tilingData->t`，L55）。
- `nk` / `nv`：Q/K 头数 / V 头数。
- `dk` / `dv`：Q/K 维度 / V 维度。
- `vStep`：V 分块步长。
- `sBlockNum`：S 状态分块数。
- `ubCalSize` / `ubRestBytes`：UB 计算区大小/剩余空间。
- `scale`：`scale_value` 属性。
- `hasGama` / `hasGamaK` / `hasAcceptedTokens`：可选输入标志。

核内分块：`alignK_ = Ceil(dk, 16)*16`（BF16 16B 对齐），`alignV_ = Ceil(dv, 16)*16`（kernel.h L66-67）。`BUFFER_NUM=1`（L27，无 double buffer）。`MAX_REPEAT_TIME=255`（L32）。

## D. 循环状态更新

`Process` 循环：每步 `S_t = α·Diag(αk)·S_{t-1} + β·(v_t - α·Diag(αk)·S_{t-1}·k_t)·k^T_t`，`o_t = S_t·q_t / √d`。`vStep` 控制 V 维度分块循环。

## E. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| `blockIdx` | `tl.program_id` | 沿 batch×head 映射 |
| `sBlockNum` / `vStep` | `tl.constexpr` BLOCK_* | 静态 meta |
| `alpha=exp(g)` / `alphaK=exp(gk)` | `tl.exp` | 门控计算 |
| `S_t = α·Diag(αk)·S_{t-1} + β·(v - α·Diag(αk)·S_{t-1}·k)·k^T` | `tl.dot` / `tl.where` / `tl.sum` | 核心循环 |
| `o_t = S_t·q·scale` | `tl.dot` | 输出 |
| BUFFER_NUM=1 / 无 double buffer | 单 buffer | 可迁移 |
| AIV-only | 单 program family | 映射 |