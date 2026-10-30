# QuantLightningIndexer — Tiling / 核切分 / 模板 dispatch (S4 + S5)

## A. 算子语义

SFA（SparseFlashAttention）前处理：选出关键的稀疏 token，并对输入 query/key 进行量化实现存8算8。

**核心公式**（README.md L16-20）：
```
out = Top_k{ W @ [ ReLU((Scale_Q ⊙ Scale_K^T) ⊙ (Q^Quant @ K^Quant^T)) ] }
```

**参数**（README.md L61-67）：
- `sparse_count`: [1, 2048]，topk 选取的 block 数量
- `sparse_mode`: 0(defaultMask) / 3(rightDownCausal)
- `layout_query`: BSND / TND
- `layout_key`: BSND / TND / PA_BSND（PageAttention）
- `pre_tokens`/`next_tokens`: 仅默认值 2^63-1

## B. Tiling 参数

基本块常量（`tiling.h` L75-81）：
- `HEAD_DIM_LIMIT = 128`（D=128 固定）
- `SPARSE_LIMIT = 2048`（sparse_count 上限）
- `G_SIZE_LIMIT = 64`（arch32 N1/G 上限）
- `G_SIZE_LIMIT_950 = 24`（arch950 N1 支持 24/64）
- `BLOCK_SIZE_LIMIT = 1024`（PA block_size 上限）
- `BLOCK_SIZE_FACTOR = 16`（PA block_size 16 对齐）

TilingData 字段（`tiling.h` L84-95）：
```
bSize, n2Size, gSize, s1Size, s2Size, sparseCount, usedCoreNum, blockSize, maxBlockNumPerBatch, sparseMode
```

## C. 模板 dispatch（`template_tiling_key.h`）

6 个模板参数：`DT_Q, DT_K, DT_OUT, PAGE_ATTENTION, Q_LAYOUT_T, K_LAYOUT_T`

**arch32（ascend910b/910_93）**：int8 输入，2 组合：
- `int8, int8, int32, PAGE_ATTENTION=1, Q_LAYOUT=BSND/TND, K_LAYOUT=PA_BSND`（PA 场景）
- `int8, int8, int32, PAGE_ATTENTION=0, Q_LAYOUT=BSND, K_LAYOUT=BSND`（非 PA BSND）
- `int8, int8, int32, PAGE_ATTENTION=0, Q_LAYOUT=TND, K_LAYOUT=TND`（非 PA TND）

**arch35（ascend950）**：fp8（e4m3fn/hifloat8）输入，6 组合（同上有 ×2 dtype）

kernel 入口 `quant_lightning_indexer<DT_Q,DT_K,DT_OUT,PAGE_ATTENTION,Q_LAYOUT_T,K_LAYOUT_T>`（L39-62），invoke `QLIPreload`。

## D. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 |
|---|---|
| KERNEL_TYPE_MIX_AIC_1_2 | grid 1D × 2 |
| S1 分核 | program_id(0) 映射到 S1 块 |
| S2 不切分 | S2 整块作为 BLOCK_N |
| Q·K^T matmul（int8/fp8） | tl.dot(反量化后 FP32) |
| dequant_scale 反量化 | tl.load(scale) * tl.load(q) |
| ReLU | tl.maximum(0, x) |
| W @ 相关性 | tl.dot(weights, scores) |
| topk | tl.topk |
| PA_BSND block_table | block_table 索引查找 |