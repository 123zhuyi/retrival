# kv_quant_sparse_flash_attention - 流水编排分析

来源：`op_kernel/kv_quant_sparse_flash_attention.cpp` + `service_cube_mla.h` + `service_vector_mla.h` + README。KV Quant SFA 是 Sparse FlashAttention 的 KV 量化版本（推理场景，离散取数 + KV dequant）。

## 1. 核心计算流程

```
Q (BF16/FP16, D=576 nope+rope)
├── sparse_indices → 离散取 KV (INT8/FP8, D=656: nope+rope+dequant_scale)
│   ├── K dequant (per-tile-128) → FP32 → K^T
│   └── V dequant (per-tile-128) → FP32
├── Bmm1: Q · K^T (MLA-absorb, D=576)
├── sparse_mode=3 mask + scale
├── FlashSoftmax (在线 max/sum/exp rescale)
├── Bmm2: P · V
└── rescale(÷softmax_sum) → attention_out (BF16/FP16)
```

## 2. 离散取数设计（sparse_indices 驱动）

sparse_indices 为 [B, Q_S, KV_N, sparse_size] 指定每行 KV 的离散位置：
- 每行有效值在前半部分，无效值在后半部分（READ ME L42-43）
- sparse_block_size ∈ [1,16]，2 幂次
- 通过 `tl.load(sparse_indices)` 查表 → 计算物理 KV 偏移 → 间接 `tl.load` KV

## 3. KV 反量化设计（Per-Token-Head-Tile-128）

量化参数和数据混合存放（quant_scale_repo_mode=1）：
- KV D 维 = nope(512) + rope(64×2) + dequant_scale(4×4) = 656
- 反量化：`K_fp32 = cast(K_int8, fp32) * dequant_scale[tile_id]`
- per-tile: tile_size=128，每个 tile 共享一个 scale

## 4. MLA-absorb 模式（attention_mode=2）

Q/K 的 D 维包含 nope 和 rope 两部分：
- Q: D=576 = nope(512) + rope(64)
- K: D=656 = nope(512) + rope(64×2) + dequant_scale(4×4)
- 分段 dot：`s = tl.dot(q_nope, k_nope.T) + tl.dot(q_rope, k_rope.T)`

## 5. 阶段 → Triton 抽象

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移 |
|---|---|---|---|
| sparse KV gather | sparse_indices + DataCopy | `tl.load(indices)` → 间接 `tl.load` | 是（语义） |
| KV dequant per-tile | Cast + Mul + tile 128 对齐 | `cast(int8/fp8, fp32) * scale` | 是（语义，精度 needs_verification） |
| Bmm1 QK (MLA) | Cube MLA-absorb | `tl.dot(q_nope, k_nope.T) + tl.dot(q_rope, k_rope.T)` | 是（数学） |
| mask + scale | sparse_mode=3 rightDownCausal | `tl.where` | 是 |
| FlashSoftmax | 在线 max/sum/exp rescale | running max/sum + `exp(m_old-m_new)` | 是（算法） |
| Bmm2 PV | Cube | `tl.dot(p, v)` | 是（数学） |
| rescale 写回 | ÷softmax_sum | `o = acc / l_i` | 是 |
| INT8→FP32 cast | Cast API | `tl.to(tl.float32)` | 是 |
| MTE DataCopy | `DataCopyPad` | `tl.load` | 否（机制） |
| Cube/Vector 同步 | 手动 sync | 不迁移 | 否（机制） |