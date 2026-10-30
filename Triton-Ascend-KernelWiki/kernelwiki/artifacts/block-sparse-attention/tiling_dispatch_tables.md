# block_sparse_attention - tiling 与 dispatch 分析

来源：`attention/block_sparse_attention/`（commit `d28d8f9...`）。内部代号 RFA（Rain Fusion Attention）。本表补 dispatch 结构、tiling/分核与 Triton 映射；块稀疏 mask 压缩方法见 [[technique-block-sparse-mask-gather]]，流水见 [[technique-flash-attention-pipeline]]（同 QK->online softmax->PV 骨架）。

## A. 架构与模板分发

`def.cpp` L75-76 注册 `ascend910b` + `ascend910_93`（A2/A3，产品支持见 `docs/aclnnBlockSparseAttention.md` L5-12：A2 √、A3 √，950PR/950DT/310P/训练系列 ×）。`attn_infra/arch/arch.hpp` 仅实现 `AtlasA2`（UB 192KB / L1 512KB / L0A 64KB / L0B 64KB / L0C 128KB）；kernel 用 `__DAV_C220_CUBE__`/`__DAV_C220_VEC__` 编译期切分（`block_sparse_attention_kernel.h` L325,331,359,...）。

任务类型：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`（`block_sparse_attention.cpp` L23）-> AIC(cube):AIV(vector) = 1:2，Atlas A2 C220。

模板参数（`block_sparse_attention_interface.cpp` L99-110 实例化 `BlockSparseAttentionKernel<...>`）：
- `BlockMmadQK` / `BlockMmadPV`：QK / PV 块 GEMM（`attn_infra/gemm/block/`）。
- `EpilogueOnlineSoftmax` / `EpilogueRescaleO`：在线 softmax / O rescale（fp32 或 fp16 low_prec 两个特化）。
- `PAGED_CACHE_FLAG`：恒为 `false`（无 paged-cache tiling key）。
- `QUERY_LAYOUT` / `KV_CACHE_LAYOUT`：0=TND，1=BNSD。

tiling key 分派（预处理 `#if TILING_KEY_VAR == <KEY>`，`block_sparse_attention.cpp` L29-94），共 **12 个 key**。base `RFA_BASE_TILING = 9000000000000000`（`tilingkey.h` L28）。key 由 `GenerateTilingKey`（`tiling.cpp` L683-748）按位叠加：dtype(BF16 +22220) / KV-layout(TND +30000000, BNSD +50000000) / softmax-prec(Half +100000) / softmaxLseFlag(+100000000) / Q-layout(TND +2, BNSD +3)。**D(64/128) 不进 key**（仅校验，`tiling.cpp` L232-236）；**blockShape 不进 key**；maskType==3 / pagedCache 分支为死代码（上游已拒绝）。

| 分发维度 | 取值 | 说明 |
|---|---|---|
| dtype | FP16 / BF16 | BF16 仅配 FLOATSM（`tiling.cpp` L513-517） |
| Q/KV layout | TND-TND / BNSD-BNSD | 必须一致（`aclnn` L135-143） |
| softmax 精度 | FloatSM(fp32) / HalfSM(fp16) | HalfSM 仅 FP16，对应 innerPrecise=0/1 |
| softmaxLseFlag | NONE / OUT_ONLY | 0=不输出 lse，1=输出 |

12 key = (2 dtype × 2 layout × {FloatSM, [FP16 才有 HalfSM]}) × {无LSE, 有LSE}。dtype×layout×prec 的合法组合 6 个（BF16 无 HalfSM），×2(LSE) = 12。

模板实参映射：`<InputDtype, SoftmaxDtype, LseMode, QueryLayout, KvCacheLayout>`。FLOATSM->`SoftmaxDtype=float`，HALFSM->`half`；`_LSE_OUT`->`LseMode::OUT_ONLY` 否则 `NONE`；TND->(0,0)，BNSD->(1,1)。

## B. 核内/核间切分

- 逻辑轴：B / N1(numHeads) / N2(kvHeads) / Q-block。**KV(S2) 不核间切分**（无 FlashDecoding），仅核内沿 selectIdx 稀疏遍历。
- 任务单元 = (Q-block × head)。`curTaskNum = GetQBlocks(qseqlen, blockShapeX) * numHeads`（`tiling.cpp` L550-553）；`GetQBlocks` 把 blockShapeX 再切成 128 粒度的 Q 子块（L103-110，`BASIC_BLOCK_SIZE=128`）。
- 核间：`blockDim = min(aicNum, totalTaskNum)`（`tiling.cpp` L601），任务沿 Q-block×heads 轴分给 AIC 核。`totalTaskNum` 为各 batch 累加；`firstBatchTaskNum`/`firstQBlockNum` 记 batch-0。
- GQA：`groupSize = numHeads / kvHeads`（L572）；约束 N1≥N2 && N1%N2==0。每个 KV head 被 groupSize 个 Q head 共享（kernel 内 `qHeadIdx = kvHeadIdx*groupSize + ...`，`block_sparse_attention_kernel.h` L506-507）。
- AIV 子核（用于 Mask2Idx）：`totalTaskNumMask = batch*numHeads*maxQBlockNum`；`avgRowPerSubCore = CeilDiv(totalTaskNumMask, blockDim*2)`；`preActivateSubCoreNum = CeilDiv(totalTaskNumMask, avgRowPerSubCore)`（`tiling.cpp` L626-628）。
- 基本块：`BASIC_BLOCK_SIZE=128`、`Q_TILE_CEIL=128`、blockShape 默认 `[128,128]`、blockShapeY 必须 128 倍数（`tiling.cpp` L396-400）。
- tile shape（`interface.cpp` L68-79）：QK `GemmShape<128,128,128>`（L1/L0）；PV L1 `GemmShape<128,128,256>`、L0 `<128,128,128>`。
- 非对齐：qSeqlen/kvSeqlen 不必被 blockShape 整除，ceilDiv 向上取整（README L504）；GEMM 内 tail 处理 `yAcutal = (last block && kvSeqlen%y!=0) ? (kvSeqlen - y*currentYIdx) : y`（`block_mmad_qk.hpp` L200-201）。
- workspace（`CalculateWorkSpace` L605-630）：三缓冲（`PRE_LAUNCH=2` -> 3 路，`WORKSPACE_BLOCK_SIZE_DB=131072`）。分区 `[gS(mm1Out) | gP(smOnline) | gOTmp(mm2) | gOUpdate(update) | gSelectNumIdx | gSelectIdx]` + 256-uint32 sync buffer。`mm1Out=blockDim*131072*sizeof(float)*3` 等。`selectIdx/selectNumIdx` 由 `Mask2IdxAndCount` 在 workspace 内派生。

## C. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| `blockDim`/core id（Q-block×head 切分） | `grid` + `tl.program_id` | 任务 = (batch, head, q-block) -> program 维 |
| 无 FlashDecoding / S2 不切核 | 单维 program（不额外切 KV） | KV 在核内循环 |
| BASIC_BLOCK_SIZE=128 / blockShape | `tl.constexpr`（BLOCK_M=128，值 source-reported） | 静态 meta |
| QK/PV 块 mmad（L1/L0A/L0B/L0C） | `tl.dot`（编译器选 buffer） | tile 级 GEMM |
| **blockSparseMask -> selectIdx 压缩** | mask 预处理：`tl.load(mask)` -> `tl.argwhere`/压缩得选中 KV-block 索引 | 见 [[technique-block-sparse-mask-gather]] |
| 稀疏 KV gather（沿 selectIdx load K/V） | `idx=tl.load(selectIdx); tl.load(k_ptr + idx*...)` | 间接 load |
| 全 0 行跳过（curSelectNum==0 continue） | `if valid_count==0: continue` | 整 Q-block 跳过 |
| 在线 softmax（running max/sum + rescale） | `tl.max`/`tl.sum` + `exp(m_old-m_new)` rescale | transferable |
| FP32 累加（FLOATSM）/ FP16（HALFSM） | `tl.dot(..., out_dtype=tl.float32)` / `tl.float16`；`if constexpr` | innerPrecise 分支 |
| softmaxLse 输出（OUT_ONLY） | `tl.store(lse_ptr, ...)` | 可选输出 |
| GQA groupSize 共享 KV | Q head 维循环内复用同 KV | 广播 |
| scale_value | `s *= scale` | transferable |
| CrossCoreSetFlag/WaitFlag（qkReady/softmaxReady/pvReady，3 event） | 不迁移（CANN 手动同步） | 抽象为 tile 依赖 |
| PreloadPipeline 3-slot（PRE_LAUNCH=2） | 不迁移（CANN 流水机制） | 抽象为 tile 复用 / fused load-store |
| MTE DataCopy GM↔L1/L0/UB、ND↔NZ | 不迁移 | Triton load/store + layout（编译器） |
| TQue/LocalTensor/BufferID | 不迁移 | Triton 虚拟内存模型 |
| KERNEL_TYPE_MIX_AIC_1_2（CV 1:2） | 不迁移（CANN 算力配比） | Triton 单 program 模型 |
