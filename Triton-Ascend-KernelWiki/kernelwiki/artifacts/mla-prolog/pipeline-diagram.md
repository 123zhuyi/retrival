# mla_prolog - 流水编排分析

来源：`docs/MlaProlog算子设计介绍.md` L9-146 + `op_kernel/kernel_mla_prolog_split_n.h`。

## 1. 四路数据流

```
token_x
  ├── ×W^DQ (MatmulCq) ──► RmsNormCq ──┬── ×[W^UQ|W^QR] (MatmulQcQr) ──┬── split(qC) ──► ×W^UK (MatmulQn) ──► q^N (queryOut)
  │                                     │                                └── split(qR) ──► RoPE ──► q^R (queryRopeOut)
  │                                     └── (RmsNorm 结果拆分)
  ├── ×[W^DKV|W^KR] (MatmulCkvKr) ──┬── split(cKV) ──► RmsNormCkv ──► Cache ──► k^C (kvCacheOut)
  │                                  └── split(kR) ──► RoPE ──► Cache ──► k^R (krCacheOut)
  └── (ffn 通路，预留? 未实现)
```

公式（设计介绍 L149-416）：
- `q^N = RmsNorm(x·W^DQ)·W^UQ·W^UK`（数学 3-matmul 展开，W^UQ·W^UK 可吸收折叠）
- `q^R = ROPE(RmsNorm(x·W^DQ)·W^QR)`
- `k^R = ROPE(x·W^KR)`
- `c^KV = RmsNorm(x·W^DKV)`

W^UQ 与 W^QR 横向拼接为 `weight_uq_qr`（公式 11），W^DKV 与 W^KR 横向拼接为 `weight_dkv_kr`（公式 7）。

## 2. 主循环 CV 流水（设计介绍 Process 伪代码 L83-146）

```
for each step (M 轴 chunk):
  if ASCEND_IS_AIC:
    MatmulCq                          # Cube: x·W^DQ
    CrossCoreSetFlag(SYNC_MMCQ_NORMSCQ_FLG)  → 通知 Vector
    MatmulCkvKr                       # Cube: x·[W^DKV|W^KR]
    CrossCoreSetFlag(SYNC_MMCKVKR_NORMROPE_FLG)  → 通知 Vector
    CrossCoreWaitFlag(SYNC_MMCQ_NORMSCQ_FLG)     ← 等 Vector 读完 normCq
    MatmulQcQr                        # Cube: cQ_norm·[W^UQ|W^QR]
    CrossCoreSetFlag(SYNC_MMQCQR_ROPEQR_FLG)     → 通知 Vector
    CrossCoreSetFlag(SYNC_ALL_CUBE_FLG)          → 全核同步
    CrossCoreWaitFlag(SYNC_ALL_CUBE_FLG)         ← 等全核
    MatmulQn                          # Cube: qC·W^UK

  if ASCEND_IS_AIV:
    GetSinCos
    CrossCoreWaitFlag(SYNC_MMCQ_NORMSCQ_FLG)     ← 等 MatmulCq
    CrossCoreSetFlag/All(SYNC_ALL_VECTOR_FLG)    → 全核同步
    RmsNormCq                         # Vector: γ·x/RMS(x)
    CrossCoreSetFlag/All(SYNC_ALL_VECTOR_FLG)    → 全核同步 (通知 Cube 可启动 MatmulQcQr)
    CrossCoreSetFlag(SYNC_MMCQ_NORMSCQ_FLG)      → 同步回 Cube
    CrossCoreWaitFlag(SYNC_MMCKVKR_NORMROPE_FLG) ← 等 MatmulCkvKr
    RmsNormRopeScatterCkvKr           # Vector: RmsNorm(cKV) + RoPE(kR) + Cache scatter
    CrossCoreWaitFlag(SYNC_MMQCQR_ROPEQR_FLG)    ← 等 MatmulQcQr
    RopeQr                            # Vector: RoPE(qR)
```

| 阶段 | 单元 | 操作 | 依赖 |
|---|---|---|---|
| 1. MatmulCq | AIC | x·W^DQ | 无 |
| 2. MatmulCkvKr | AIC | x·[W^DKV\|W^KR] | 无（与 MatmulCq 并行） |
| 3. RmsNormCq | AIV | γ·x/RMS(x) | 等 MatmulCq |
| 4. RmsNormRopeScatterCkvKr | AIV | RmsNorm(cKV) + RoPE + Cache | 等 MatmulCkvKr |
| 5. MatmulQcQr | AIC | cQ_norm·[W^UQ\|W^QR] | 等 RmsNormCq |
| 6. RopeQr | AIV | RoPE(qR) | 等 MatmulQcQr |
| 7. MatmulQn | AIC | qC·W^UK | 等全核 MatmulQcQr |

## 3. RoPE 优化公式（设计介绍 L359-415）

CANN 实现按奇偶拆分的优化 RoPE（公式 15）：`x` 拆成奇偶两半，`cos`/`sin` 连续。Triton 层可用标准复数旋转等价实现。

## 4. Cache 写回（service_scatter_cache.h，设计介绍 L437-454）

`cache_index`（INT64, [B,S]）标识每个 token 的 KV cache 更新位置。PA 场景支持 ND 和 NZ 两种存储格式。

## 5. 3 量化场景（kernel 入口 L69-99）

| 场景 | x/weight | weight_uq_qr | kv_cache/kr_cache |
|---|---|---|---|
| 非量化 | BF16 | BF16 | BF16 |
| 部分量化 KV 非量化 | BF16 | INT8 | BF16 |
| 部分量化 KV 量化 | BF16 | INT8 | INT8 (per-channel) |

## 6. 阶段 -> Triton 抽象

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移 |
|---|---|---|---|
| 4 matmul | Cube MMA + L0C 累加 | `tl.dot`（FP32 acc） | 是（数学） |
| RmsNorm | Vector sum+rsqrt | `tl.sum`+`rsqrt` | 是（数学） |
| RoPE | 奇偶分拆 cos/sin 乘 | 复数旋转 | 是（数学） |
| Cache scatter | cache_index scatter | `tl.store(cache + idx*stride)` | 是 |
| 6 CV sync event | CrossCoreSetFlag/WaitFlag | 不迁移（抽象为 tile 依赖） | 否（CANN 专有） |
| 全核同步(SYNC_ALL_CUBE/VECTOR) | CrossCore 全局同步 | 不迁移（编译器调度） | 否（CANN 专有） |
| 权重拼接(W^UQ\|W^QR) | 合并 matmul 后 split | 合并 kernel 或 2 个 dot | 是（抽象） |
| NZ 格式 cache | ND↔NZ 转换 | 不迁移（Triton layout） | 否（CANN 专有） |