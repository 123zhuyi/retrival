# lightning_indexer - tiling 与 dispatch 分析

来源：`attention/lightning_indexer/`（commit `d28d8f9...`）。稀疏注意力索引选择器。

## A. 架构与模板分发

`def.cpp` L75-76 注册 `ascend910b` + `ascend910_93`（A2/A3 训练+推理）。产品支持：A2 训练+推理 √、A3 训练+推理 √；950PR/950DT ×。

任务类型：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`（`lightning_indexer.cpp` L45），AIC:AIV=1:2。

模板参数（`lightning_indexer_template_tiling_key.h` L33-41）：DT_Q、DT_K、DT_OUT（仅 INT32）、PAGE_ATTENTION（0/1）、LAYOUT_T（BSND/TND）、K_LAYOUT_T（BSND/TND/PA_BSND）、DT_W_FLAG（0/1）。**24 合法组合**（L45-86）：

| PA | Q Layout | K Layout | DT_W_FLAG | dtype |
|---|---|---|---|---|
| 1(PA) | BSND/TND | PA_BSND | 0/1 | FP16 |
| 1(PA) | BSND/TND | PA_BSND | 0/1 | BF16 |
| 0 | BSND | BSND | 0/1 | FP16 |
| 0 | TND | TND | 0/1 | FP16 |
| 0 | BSND | BSND | 0/1 | BF16 |
| 0 | TND | TND | 0/1 | BF16 |

8 种 layout/dtype 组合 × 2 DT_W_FLAG（weights=FP32/非FP32） = 16 组合 + 8 PA 组合 = 24。

## B. 核内/核间切分

- `SplitCore`（kernel.h L279-357）：按 `totalBaseBlockNum` 计算 `minBlockPerCore`。块粒度 = (B×N2×S1g×S2) 的 base block 数。
- 三轴切分：B×N2×S1g×S2。S2 跨核切分（多个核共享同一 Q 行的 S2 循环）。第一个核处理 `isLD=true`（decode reduce）。
- 基本块：`M_BASE_SIZE=512`、`S2_BASE_SIZE=512`、`HEAD_DIM=128`（kernel.h L84-86）。
- workspace 双缓冲：`mm1ResGm`（存 S 分数）、`vec1ResGm`（LD 中间结果）、`vec1ParamGm`（LD 参数）。

## C. Triton-Ascend 抽象映射

| CANN | Triton | 说明 |
|---|---|---|
| core id（B×N2×S1g×S2 切分） | `tl.program_id` + `grid` | 四轴映射 |
| Q·K^T（ComputeMm1） | `tl.dot(q, k.trans)` | GEMM |
| ReLU + weights 缩放 | `tl.maximum(s,0) * weight` | elementwise |
| topK 选择 | `tl.argsort` 或等效 | 选 topK 索引 |
| PageAttention（blockTable） | `tl.load(blockTable)` → 间接 load | PA 路径 |
| sparse_indices 输出 | `tl.store(indices_ptr, idx)` | INT32 输出 |
| sparse_values 输出 | `tl.store(values_ptr, val)` | 可选输出 |
| M_BASE_SIZE=512 | `tl.constexpr` | 静态 meta |
| CrossCore flag | 不迁移 | CANN 手动同步 |
| MTE DataCopy / ND↔NZ | 不迁移 | Triton load/store |