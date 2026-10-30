# block_sparse_attention_grad - tiling 与 dispatch 分析

来源：`attention/block_sparse_attention_grad/`（commit `d28d8f9...`）。前向分析见 `artifacts/block-sparse-attention/tiling_dispatch_tables.md`。本表聚焦反向与前向的差异。

## A. 架构与模板分发

`def.cpp` L84-85 注册 `ascend910b` + `ascend910_93`（同前向）。产品支持（docs L5-14）：**仅训练**--Atlas A2 训练 √、Atlas A3 训练 √；A2 推理 ×、A3 推理 ×；950/310P/其余 ×。（前向 A2/A3 训练+推理均 √；反向训练专用。）

任务类型：grad 入口 `block_sparse_attention_grad.cpp` **未显式** `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`（前向 L23 有），但 `vecCoreIdx/2`（kernel.h L537）与 AIC/AIV 特化暗示 1:2 `[needs_verification]`。

模板参数（`block_sparse_attention_grad_interface.h` L123-125 实例化 `BlockSparseAttentionGradKernel<...>`）：
- 3 个 cube GEMM 模块 `BlockMmadBSAG{1,2,3}`（均用 `MmadAtlasA2SBSAG2` 策略，interface.h L58/73/88）。
- 4 个 vec epilogue 模块：`EpilogueFAGPre`（清零）、`EpilogueFAGSfmg`（SoftmaxGrad=D）、`EpilogueFAGOp`（SimpltSoftmax=重算P+CalDs）、`EpilogueFAGPost`（rescale+写回）。
- `INPUT_LAYOUT`：0=TND，1=BNSD。

tiling key 分派（`block_sparse_attention_grad.cpp` L44-60，`TILING_KEY_IS(0|1|10|11)`），共 **4 个 key**（无独立 tilingkey.h）：

| key | dtype | layout | 说明 |
|---|---|---|---|
| 0 | FP16 | TND | half, InputLayout=0 |
| 1 | FP16 | BNSD | half, InputLayout=1 |
| 10 | BF16 | TND | bfloat16_t, InputLayout=0 |
| 11 | BF16 | BNSD | bfloat16_t, InputLayout=1 |

仅按 **dtype × Q/KV-layout** 分派。**无** innerPrecise/softmaxLseFlag/maskType/pagedCache 轴（前向 12 key 编码 6 轴，base `RFA_BASE_TILING=9000000000000000`；反向**抛弃** RFA 方案，`GenerateTilingKey` tiling.cpp L497-514）。反向 softmax 内部恒 FP32（无前向 FLOATSM/HALFSM 之分）；Lse 恒作为输入消费（无 LseMode）。

## B. 核内/核间切分

- 逻辑轴：B / N1(numHeads) / N2(kvHeads) / Q-block。**KV(S2) 不核间切分**（无 FlashDecoding），核内沿 KV 块遍历。
- 任务单元 = (Q-block × head)：`totalTaskNum = Σ_b qBlocks_b · numHeads`（tiling.cpp L404-405），同前向轴。
- 核间：`blockDim = min(aicNum, totalTaskNum)`（L418），用 **AIC 核**；`taskNumPerCore = totalTaskNum/coreNum`，`tailTaskNum = totalTaskNum%coreNum`（L312-313）。
- **每核任务描述子数组**（与前向不同）：`beginBatch/beginHead/beginQSeqOffset/preQSeqLengths/preKVSeqLengths[64]`（tiling.h L65-69）。前向用 sub-core（`avgRowPerSubCore`/`preActivateSubCoreNum`）分区。
- 遍历序：TND `Batch->SeqBlock->Head`（L323-339）；BNSD `Batch->Head->SeqBlock`（L341-355）。
- GQA：`groupSize = numHeads/kvHeads`（kernel.h L302,543），KV head 每 groupSize 个 Q head 推进一次（L258-260）。约束 N1≥N2 && N1%N2==0。
- 基本块：`BASIC_BLOCK_SIZE=128`、`basicQBlockSize=basicKVBlockSize=128`（tiling.cpp L68,467-468）；blockShape 默认 `[128,128]`。
- 非对齐：ceilDiv + GEMM tail（同前向）。
- **workspace（FP32）**（`CalculateWorkSpace` L422-447）：`[gS(sOutSize,重算S) | gDp(dPOutSize,dP=dOut·V^T) | gDq(dQOutSize,FP32累加) | gDk(dKOutSize) | gDv(dVOutSize) | sftmg(gradSize,softmax-grad)]`。`WORKSPACE_BLOCK_SIZE_DB=128*128*2*2`（L69）。**无 selectIdx/selectNumIdx**（前向有，用于 CSR gather）。
- softmax-grad tiling：`AscendC::SoftMaxGradTilingFunc`（L278-300）。
- UB：`postUbBaseSize = (ubSize - sizeof(TilingData) - 2KB)/3/256·256`（L270-276）。

## C. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| `blockDim`/core id（Q-block×head） | `grid` + `tl.program_id` | 任务=(batch,head,Q-block) -> program 维 |
| 每核 begin 数组 | program_id 直接算 (batch,head,qblk) | 无需 host 预算 begin |
| 无 FlashDecoding / S2 不切核 | 单维 program（不额外切 KV） | KV 核内循环 |
| BASIC_BLOCK_SIZE=128 / blockShape | `tl.constexpr`（BLOCK_M=128） | 静态 meta |
| **mask 门控稠密扫描**（`if(mask) skip`） | `m=tl.load(mask_row); tl.where(m, score, -inf)` 或跳过 | 见 [[kernel-block-sparse-attention-grad]] §3 |
| 重算 P=exp(S·scale−Lse) | `p=tl.exp(tl.dot(q,k)*scale - tl.load(lse))` | 用前向保存 Lse |
| 3 cube GEMM（S&dP / dQ / dK&dV） | `tl.dot`（编译器选 buffer） | tile 级 GEMM |
| dQ/dK/dV AtomicAdd 跨 KV 块累加 | `tl.atomic_add` 或 per-program FP32 accum 后写 | 反向需累加 |
| softmax-grad D=Σ(dY·O) | `tl.sum(dY*O, axis=-1)` | reduction |
| dS=P·(dP−D) | `p*(dp - D[:,None])` | elementwise |
| rescale（dQ/dK*=scale，dV 不） | `dq*=scale; dk*=scale` | 注意 dV 不乘 scale |
| scale_value | `*= scale` | transferable |
| CrossCore flag（CUBE2VEC/VEC2CUBE/CUBE2POST, id 7/8/9） | 不迁移（CANN 手动同步） | 抽象为 tile 依赖 |
| preTaskInfo 1-deep delay | 不迁移（CANN 流水机制） | 抽象为 tile 复用 |
| MTE DataCopy / ND↔NZ / TQue / AtomicAdd(CANN) | 不迁移 | `tl.load/store`/`tl.atomic_add`/layout |
| SoftmaxGradFront/SoftMaxGradTilingFunc | 不迁移（CANN intrinsic） | 手写 reduction |
