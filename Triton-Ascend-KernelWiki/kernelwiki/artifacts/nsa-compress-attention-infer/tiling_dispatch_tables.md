# nsa_compress_attention_infer - tiling 与 dispatch 分析

来源：`attention/nsa_compress_attention_infer/`（commit `d28d8f9...`）。NSA（Native Sparse Attention）推理流水线的第一阶段：压缩注意力 QK^T + softmax + PV + 滑动窗口重要性分数 + topK。

## A. 架构与模板分发

`def.cpp` L88-89 注册 `ascend910b` + `ascend910_93`（A2/A3 训练+推理）。产品支持（docs L5-12）：A2 训练+推理 √、A3 训练+推理 √；950PR/950DT ×、200I/500 A2 ×。注意：`block_sparse_attention`（前向）也是 A2/A3 训练+推理；`block_sparse_attention_grad` 仅训练。

任务类型：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`（`nsa_compress_attention_infer.cpp` L32），AIC(cube):AIV(vector)=1:2，Atlas A2 C220。

模板参数（`ncai_public_define.h` L46-52 `NCAIType<Q_T, KV_T, OUT_T, LAYOUT>`；kernel 分两套独立实现）：
- `NsaCompressAttentionInferAic`（Cube 路径，`.h` L40-689）：`ProcessMm1`（QK^T）→ cross-core sync → `ProcessMm2`（PV）。
- `NsaCompressAttentionInferAiv`（Vector 路径，`.h` L690-1565）：`ProcessSoftmax` → `ProcessImportanceScore` → `ProcessTopK`。
- 无 `attn_infra` 模板库（与 block_sparse_attention 系列不同），kernel 为 NSA 独立实现。

tiling key 分派（`nsa_compress_attention_infer.cpp` L35-54）：

| key | dtype | layout | 说明 |
|---|---|---|---|
| 0 | FP16/BF16（编译期 ORIG_DTYPE_QUERY） | TND | half, LAYOUT=TND |
| 1 | FP16/BF16 | BSND | half, LAYOUT=BSND |

仅 2 个 key，按 **layout** 分派。dtype 由编译宏 `ORIG_DTYPE_QUERY` 在构建时选择（`cpp` L34,45），tiling key 不区分 FP16/BF16。无 maskType/innerPrecise/lse/blockShape 等轴。

## B. 核内/核间切分

- 逻辑轴：B（batch）× N2（kvHeads）× Q-S（query 序列长度）。KVS2 不核间切分，核内遍历。
- `AssignQSeqLen`（tiling.cpp L313-326）：根据 Q-S 和核心数计算 `qSeqLenSplitSize`（每批每个核处理多少 query token）。
- `AssignTask`（L327-445）：分配 `processNum = qSeqLenSplitSize * kvHeadNum`，任务 = （Qtoken × kvHead）对。
-  GQA：`groupSize = numHeads / numKeyValueHeads`，KV 头共享。
- 基本块：blockSize 默认 ＜= 128（16 倍数），coressBlockSize/selectBlockSize 仅支持 16、32、48、64、80、96、112、128。
- workspace（tiling.cpp L452-455）：4 个缓冲区 - `mm1Res`（FP32 QK^T 结果）、`mm2n`（FP16/BF16 softmax 后）、`scoreIn`（FP32 重要性分数输入）、`opKIn`（FP32 topK 输入），全部 2x 双缓冲。
- 单 token（Q_S=1）vs 多 token（Q_S<=4）路径：通过 `actualQSeqLenOptional` 和 `actualCmpKvSeqLenOptional` 的有效性自动选择。

## C. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| `blockDim`/core id（B×N2×Q-S 切分） | `grid` + `tl.program_id` | 任务=（batch，kvHead，Q-token）→ program 维 |
| KV 3D `[umBlocks, blockSize, N*D]` + blockTable | `tl.load（blockTable）` → 物理 block → `tl.load` KV | 必选 PagedAttention |
| ProcessMm1（QK^T） | `tl.dot（q, k.trans）` | Cube GEMM |
| ProcessSoftmax | `tl.exp（s - max）/ sum` | 在线/简单 softmax |
| ProcessMm2（PV） | `tl.dot（p, v）` | Cube GEMM |
| ProcessImportanceScore（滑动窗口卷积） | 遍历 P_slc = sum_over 三角形权重 x score | 独特 NSA 操作 |
| ProcessTopK | `tl.argsort` 或等效 topK 选择 | 输出 int32 索引 |
| CrossCore flag（MM1_READY/SOFTMAX_READY） | 不迁移 | Cube↔Vector 依赖 |
| 双缓冲（l1kPingPongFalg 等） | 不迁移 | CANN 流水机制 |
| MT E DataCopy / ND↔NZ | 不迁移 | `tl.load`/`tl.store`+ layout |