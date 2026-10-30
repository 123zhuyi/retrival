# mla_preprocess - 流水编排分析

来源：`README.md` L10-54 + `op_kernel/mla_preprocess.cpp` + `op_kernel/mla_preprocess_{fp16,bf16}.h`。`*_iterator.h` 与 ProcessCube/ProcessVector 的 CV 分离属 CANN 专有，抽象为 fused 计算。

## 1. 数据流（MLA 前处理融合）

按 README L12-19 的 6 通路数据流：

```
input x
  │
  ▼ RmsNormQuant₁ (γ₀/β₀/quantScale₀/quantOffset₀)
  │
  ▼ mm1: ×W^DQKV  (降维, Cube)  -> 拆分 ┐
  │                                     │
  ├─通路1 (Q) ──► RmsNormQuant₂ ─► mm2: ×W^UQ (Cube) ──┬─通路3 ─► mm3: ×W^UK (Cube) ─► q^N (qOut)
  │                                                    └─通路4 ─► RoPE ─────────────► q^R (qRopeOut)
  │
  └─通路2 (K) ──┬─通路5 ─► RmsNorm₃ ─► Cache (slot_mapping) ─► k^N (kvCacheOut)
                └─通路6 ─► RoPE ─────► Cache (slot_mapping) ─► k^R (krCacheOut)
```

公式（README L20-54）：
- `RMS(x)=sqrt(mean(x²)+eps)`，`RmsNorm(x)=γ·x/RMS(x)`，`RmsNormQuant(x)=(RmsNorm(x)+bias)·deqScale`
- `q^N = RmsNormQuant(x)·W^DQKV·W^UK`（注：公式为 2 matmul 形式，反映 MLA 权重吸收 W^UQ·W^UK 可折叠；kernel 实际实现 3 个 matmul mm1/mm2/mm3，吸收与否 `needs_verification`）
- `q^R = ROPE(x^Q)`
- `k^N = Cache(RmsNorm(RmsNormQuant(x)))`
- `k^R = Cache(ROPE(RmsNormQuant(x)))`

## 2. CV 阶段（ProcessCube + ProcessVector）

kernel 入口（`mla_preprocess.cpp` L52-53）先 `ProcessCube()` 再 `ProcessVector()`：

| 阶段 | 单元 | 操作 | tiling 方法 |
|---|---|---|---|
| Cube | AIC | 3 个 matmul（mm1 W^DQKV / mm2 W^UQ / mm3 W^UK）+ RoPE+Concat | `RopeConcatTiling` |
| Vector | AIV | 2 次 RmsNormQuant + RmsNorm₃ + EinSumQuant + Cache 写回 | `RmsNormQuantTiling` / `EinSumQuantTiling` |

注：实际 Cube/Vector 内部子阶段排布（哪个 matmul 与哪个 norm 衔接、是否有 double buffer）在 `mla_preprocess_{fp16,bf16}.h` 内，未逐行确认，`needs_verification`。

## 3. buffer 搬运（CANN 专有，不迁移）

`op_kernel/*_iterator.h` 封装 CANN 各级 buffer 间搬运：

| iterator | 通路 |
|---|---|
| `gm_to_l1_iterator.h` | GM -> L1 |
| `gm_to_ub_iterator.h` | GM -> UB |
| `l1_to_l0_iterator.h` | L1 -> L0A/L0B |
| `l0c_to_gm_iterator.h` / `l0c_to_l1_iterator.h` / `l0c_to_ub_iterator.h` | L0C -> GM/L1/UB |
| `l1_to_ub_iterator.h` / `l1_to_bt_iterator.h` / `l1_to_fb_iterator.h` | L1 -> UB/BiasTable/FormatBuffer |

这些是 CANN 手动 buffer 搬运 + ND↔NZ 格式转换机制，Triton 层由编译器管理 layout，**不迁移**；抽象为 tile 级 load/store + fused 计算。

## 4. workspace 布局

3 个等大 section（s1Gm/s2Gm/s3Gm，各 `maxWorkspaceSize`，kernel 入口 L41-43），存放 matmul/RmsNorm 中间结果。`__DAV_C220_CUBE__` guard（L36）确认 A2/A3 Cube 路径。

## 5. Cache 写回（scatter store）

K 通路结果通过 `slot_mapping`（INT32，shape [tokenNum]）索引写入 `kv_cache`/`kv_cache_rope`（PagedAttention 风格 block cache）。cacheMode 决定 cache shape/格式（README L208-220）：
- cacheMode=0：kvCache [blockNum,blockSize,1,576]（k^N+k^R 拼接），ND。
- cacheMode=1：拆分 kvCacheOut[...,512] + krCacheOut[...,64]，ND。
- cacheMode=2：krope/ctkv 转 NZ + per_head 对称量化 int8。
- cacheMode=3：krope/ctkv 转 NZ。

## 6. 阶段 -> Triton 抽象

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移 |
|---|---|---|---|
| RmsNormQuant | Vector `tl.sum`+rsqrt+cast | FP32 reduce + 量化 cast | 是（数学） |
| mm1/mm2/mm3 | Cube MMA + L0C 累加 | `tl.dot`（FP32 acc） | 是（数学） |
| RoPE | Vector 旋转乘 | 复数旋转 | 是（数学） |
| Cache 写回 | slot_mapping scatter + ND↔NZ | `tl.store(cache + slot*stride)` | 是（scatter）；NZ 不迁移 |
| buffer 搬运/迭代器 | GM/L1/L0C/UB 手动 | load/store（编译器） | 否（机制） |
| CV 分离/ProcessCube+Vector | AIC/AIV 分相 | fused 单 program | 否（机制，抽象） |
| swizzle/ping-pong/shuffleK | CANN 调度 | 不迁移 | 否（机制） |
