---
id: technique-block-sparse-mask-gather
title: Block-sparse mask compression to CSR + sparse KV gather
type: technique
tags: [block_sparse_attention, sparse_attention, mask_compression, kv_gather, csr, online_softmax]
techniques: [sparse_kv_gather, sparse_attention_tiling, fp32_accumulation]
architectures: [Ascend910B1, Ascend910_93]
hardware_features: [unified_buffer, cube_vector_1_to_2, mte2, mte3, buffer_id_sync]
confidence: inferred
reproducibility: snippet
prerequisites: [了解 FlashAttention 的 QK->在线 softmax->PV 流程, 块稀疏注意力与 blockSparseMask 概念, Ascend cube/vector 协同与 MTE 搬运基本概念]
related: [kernel-block-sparse-attention, technique-sparse-kv-gather, technique-flash-attention-pipeline, technique-sparse-attention-pipeline]
artifact_dir: artifacts/block-sparse-attention
sources: [ops-transformer-block-sparse-attention]
---

# 块稀疏 mask 压缩为 CSR + 稀疏 KV gather

把块级稀疏注意力的 INT8 `blockSparseMask` 先压缩成 CSR 式（选中索引 + 每行计数），再让 GEMM 仅 gather 加载被选中的 KV 块，从而获得真正的计算稀疏（masked 块零开销），而非"遍历所有块再置 -inf"。来源：`attention/block_sparse_attention/`（commit `d28d8f9...`），`op_kernel/block_sparse_attention_kernel.h` L93-248（`Mask2IdxAndCount`）+ `attn_infra/gemm/block/block_mmad_qk.hpp` L152-227 / `block_mmad_pv.hpp` L121-186。用于 [[kernel-block-sparse-attention]]。

## 1. 问题与动机

块稀疏注意力用 `blockSparseMask` `[B, N, qBlockNum, kvBlockNum]`（INT8，1=该 Q 块对该 KV 块计算，0=跳过）描述稀疏 pattern。朴素实现是遍历所有 KV 块、对 mask=0 的块把 score 置 `-inf`--仍要加载 K/V 并做 QK/PV，只是结果被屏蔽，**没有节省算力与带宽**。本方法把 mask 预处理成"每行选中的 KV-block 索引列表"，GEMM 只遍历该列表，未选中块完全不加载、不计算。

与 [[technique-sparse-kv-gather]] 的区别：SFA 由外部选择算法给出 `sparse_indices`（已是最小索引集，离散 token 级），本方法从**稠密 4D 掩码**自行压缩出块级索引，pattern 来源不同、压缩发生在算子内部。

## 2. CANN 实现（AtlasA2，cube:vector=1:2）

**Stage0 mask 压缩（vector 核，全局一次）** — `Mask2IdxAndCount`（kernel.h L93-248）：
- 行空间 = `batch * numHeads * maxQBlockNum`（每行 = 一个 (batch, head, Q-block)），按 `PRE_ROW_TILE=128 × PRE_COL_TILE=64` 分块（L96-97,141）。
- 每 sub-core 处理 `avgRowPerSubCore` 行（tail 由 `preActivateSubCoreNum` 收尾）。
- 流水：`DataCopyPad` 读 int8 mask（L162-171）-> `Cast` uint8->half->float（L175-182）-> `CompareScalar ... GE 1.0` 产生位掩码（L189-192）-> `GatherMask` 把选中 KV-block 索引紧凑写入 `selectIdx`（int32）并产出每行计数 `selectNumIdx`（int32）（L197-202,208-234）。
- `idxPingPongFlag`/`countPingPongFlag` 双缓冲；HardEvent 编排 V_MTE2/MTE2_V/MTE3_V/S_MTE3（L130-137）。
- `SyncAll<false>` 屏障（L328-329）保证 `selectIdx`/`selectNumIdx` 全核可见后再进 cube。

**Stage1-4 GEMM 沿 selectIdx gather** — `BlockMmadQK::operator()`（block_mmad_qk.hpp L152-227）/ `BlockMmadPV::operator()`（block_mmad_pv.hpp L121-186）：
- 每个任务（batch, head, Q-block）先读 `curSelectNum = gSelectNumIdx.GetValue(curSelectIdx)`（kernel.h L510）；**`==0` 则 `continue` 跳过整 Q-block**（L512-514）。
- 沿选中索引遍历：`currentYIdx = gSelectIdx.GetValue(currentSelectYIdx)`（L194/152），`offsetInKV = currentYIdx * y + currentYoffset`（L195），仅 `copyGmToL1B` 加载该块的 K/V（L212/170）。
- 循环守卫 `processSize < nActual && currentSelectYIdx < selectNum && currentYIdx < kvYBlockNum && offsetInKV < kvSeqlen`（L199）；一个 Y-block 耗尽即 `currentSelectYIdx++` 读下一个选中索引（L218-226）。未选中块永不加载。
- 4 阶段 cube↔vector 流水（QK -> 在线 softmax -> PV -> rescale）见 [[technique-flash-attention-pipeline]]，3 个 cross-core flag 协调（qkReady/softmaxReady/pvReady）。

## 3. 可迁移要点（transferable）

- **mask -> CSR 压缩**：稠密块掩码 -> 变长选中索引列表 + 计数。语义等价 `argwhere(mask==1)`，是任何块稀疏注意力都可用的预处理。
- **gather-only 计算**：QK/PV 的内层 KV 循环改为沿索引列表间接寻址，跳过未选中块，省算力+带宽。
- **全 0 行短路**：选中数为 0 的 Q-block 直接跳过，零开销。
- **与在线 softmax 正交**：max/sum rescale 不关心 KV 块是否连续，稀疏 gather 天然兼容在线 FlashSoftmax。
- **tail/非对齐**：最后一个选中块按 `kvSeqlen` 截断（`yAcutal`，block_mmad_qk.hpp L200-201），支持 qSeqlen/kvSeqlen 不被 blockShape 整除。

## 4. Triton-Ascend 抽象

- mask 压缩建议在 **host 预处理**或**独立 kernel** 产出 `selectIdx`/`selectNum`，避免每个 attention program 重复压缩（CANN 是全局做一次，Triton 无跨核共享 workspace 的天然对应）。
- attention kernel 接收 `select_idx_ptr`/`select_num_ptr`，内层循环 `idx = tl.load(select_idx_ptr + ...); k = tl.load(k_ptr + idx*BLOCK_N + ...)`。
- 变长索引在静态 kernel 中用**上界循环 + `if s >= sel_count: break`** 或 **padding 索引 + mask** 表意（伪代码见 `artifacts/block-sparse-attention/block-sparse-attention-triton-skeleton.py`）。
- 全 0 行：`if sel_count == 0: 写零/跳过`。
- CANN 专有不迁移：`GatherMask`/`CompareScalar` 原语（-> `tl.where`/布尔索引语义等价）、`DataCopyPad`/MTE 搬运（-> `tl.load`）、CrossCore flag（-> tile 依赖）、ping-pong HardEvent（-> 编译器调度）。

## 5. 适用与边界

- 适用：块级稀疏注意力（掩码描述 pattern），尤其稀疏度较高时收益明显（跳过块比例大）。
- 不适用：token 级离散稀疏已有现成索引集（用 [[technique-sparse-kv-gather]]）；稠密注意力（无需压缩）。
- 边界 [needs_verification]：mask 压缩的并行策略与 UB 占用、变长 gather 在 Triton 静态 tile 下的最优上界对齐、压缩粒度（blockShapeX/Y 与 128 对齐）均需结合硬件实测。本页 `reproducibility: snippet`（含伪代码片段，未编译/未实测）。来源 [[ops-transformer-block-sparse-attention]]。
