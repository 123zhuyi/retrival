# mla_prolog_v3 - 流水编排分析

来源：`docs/MlaPrologV3算子设计介绍.md`（L15-143）+ `op_kernel/mla_prolog_v3.cpp`（L25-28 模板入口）。MLA Prolog V3 为 DeepSeek V2/V3 核心 MLA 的前处理融合算子，4 路 matmul + 2 RmsNorm + 2 RoPE + Cache 更新，是 attention 算子里 CV 流水同步最复杂的（6 个同步点）。

## 1. 整体计算流程（设计介绍 L15-25）

```
x (token_x)
├── x · W^{DQ}  (MatmulCq)  ──→ c^Q ──→ RmsNorm ──→ c_norm^Q
│                                               ├── c_norm^Q · W^{UQ}  (MatmulQcQr) ──→ q^C ──→ q^C · W^{UK} (MatmulQn) ──→ q^N (query)
│                                               └── c_norm^Q · W^{QR}  (MatmulQcQr) ──→ q^R ──→ RoPE ──→ q^R_rope (query_rope)
│
└── x · [W^{DKV}|W^{KR}]  (MatmulCkvKr) ──→ [c^{KV}|k^R]
                                            ├── c^{KV} ──→ RmsNorm ──→ Cache ──→ kv_cache
                                            └── k^R    ──→ RoPE ──→ Cache ──→ kr_cache
```

## 2. Cube-Vector 交替流水与 6 个同步点（设计介绍 L47-68 / L88-143）

MLA Prolog V3 的 4 个 matmul 在 Cube 侧，2 个 RmsNorm + 2 个 RoPE 在 Vector 侧，不同 matmul 的分核策略不同，导致 Cube 和 Vector 之间需要复杂的核间同步。

### 主流程伪码（设计介绍 L88-143）

```
for each token block:
    // ────── Cube 侧 ──────
    if ASCEND_IS_AIC:
        MatmulCq(x, W_dq, cq)                          # ① x·W^{DQ} -> c^Q
        CrossCoreSetFlag(SYNC_MMCQ_NORMSCQ)             # 同步点1：通知 Vector c^Q 就绪
        MatmulCkvKr(x, W_dkv_kr, ckvkr)                 # ② x·[W^{DKV}|W^{KR}] -> [c^{KV}|k^R]
        CrossCoreSetFlag(SYNC_MMCKVKR_NORMROPE)         # 同步点2：通知 Vector c^{KV}+k^R 就绪
        CrossCoreWaitFlag(SYNC_MMCQ_NORMSCQ)            # 同步点3：等待 Vector RmsNormCq 完成
        MatmulQcQr(c_norm_q, W_uq_qr, qcqr)             # ③ c_norm^Q·[W^{UQ}|W^{QR}] -> [q^C|q^R]
        CrossCoreSetFlag(SYNC_MMQCQR_ROPEQR)             # 同步点4：通知 Vector q^R 就绪
        CrossCoreSetFlag(SYNC_ALL_CUBE)                  # 同步点5：等待所有 Cube 核的 MatmulQcQr 完成
        CrossCoreWaitFlag(SYNC_ALL_CUBE)
        MatmulQn(q_c, W_uk, qn)                          # ④ q^C·W^{UK} -> q^N

    // ────── Vector 侧 ──────
    if ASCEND_IS_AIV:
        CrossCoreWaitFlag(SYNC_MMCQ_NORMSCQ)             # 等待同步点1
        RmsNormCq(cq, norm_cq)                          # RmsNorm(c^Q) -> c_norm^Q
        CrossCoreSetFlag(SYNC_ALL_VECTOR)                # 同步点（Vector 全核同步）
        CrossCoreWaitFlag(SYNC_ALL_VECTOR)
        CrossCoreSetFlag(SYNC_MMCQ_NORMSCQ)              # 通知 Cube c_norm^Q 就绪
        CrossCoreWaitFlag(SYNC_MMCKVKR_NORMROPE)         # 等待同步点2
        RmsNormRopeScatterCkvKr(ckvkr)                   # RmsNorm(c^{KV}) + RoPE(k^R) + Cache 更新
        CrossCoreWaitFlag(SYNC_MMQCQR_ROPEQR)             # 等待同步点4
        RopeQr(qr, qr_rope)                              # RoPE(q^R) -> q^R_rope
```

### 6 个同步点总览

| 编号 | 同步信号 | 发起方 | 等待方 | 说明 |
|---|---|---|---|---|
| 1 | SYNC_MMCQ_NORMSCQ | AIC | AIV | MatmulCq 完成，Vector 可开始 RmsNormCq |
| 2 | SYNC_MMCKVKR_NORMROPE | AIC | AIV | MatmulCkvKr 完成，Vector 可开始 RmsNormCkv + RoPE \\
| 3 | SYNC_MMCQ_NORMSCQ（反向） | AIV | AIC | RmsNormCq 完成，AIC 可开始 MatmulQcQr |
| 4 | SYNC_MMQCQR_ROPEQR | AIC | AIV | MatmulQcQr 完成，AIV 可开始 RopeQr |
| 5 | SYNC_ALL_CUBE | AIC | AIC | 所有 Cube 核的 MatmulQcQr 完成，才能开始 MatmulQn（分核策略不同） |
| 6 | SYNC_ALL_VECTOR | AIV | AIV | Vector 全核同步（设计介绍 L68） |

## 3. 四路 matmul 分核策略差异

| matmul | 切分轴 | 分核数 | 是否需全核同步 |
|---|---|---|---|
| MatmulCq | 沿 Hcq 按 AIC 核数 | AIC 核数 | 否 |
| MatmulCkvKr | 固定 64 块 | 64 | 否 |
| MatmulQcQr | 沿 Hqc 按 AIC 核数 | AIC 核数 | 否 |
| MatmulQn | 不切 N（M 轴单点） | 1 | 是（依赖 MatmulQcQr 全核输出） |

## 4. KVCache / KRCache 更新（设计介绍 L435-454）

```
cacheIndex = [B, S] 标识每个 token 在 cache 中的目标位置
KVCache[PA ND/NZ] 由 cacheIndex 驱动 scatter 更新
KRCache[PA ND/NZ] 同 KVCache 更新逻辑
```

PA_ND 格式：每行由 cacheIndex 映射到 block 内行位置。
PA_NZ 格式：32B 小块分形，stride = 32B * blockSize。

## 5. Triton-Ascend 抽象

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移 |
|---|---|---|---|
| 4 路 matmul | Cube 高阶 API | `tl.dot` × 4 | 是（数学） |
| 2 个 RmsNorm | AscendC Vector API | `tl.var` + `* rsqrt` + `* gamma` | 是（算法） |
| 2 个 RoPE | 偶奇拆分 + cos/sin 旋转 | 偶奇 split + `* cos` + 翻转 `* sin` | 是（算法） |
| KVCache scatter | `DataCopyPad` + cacheIndex | `tl.store(cache + idx, ...)` | 是（语义） |
| CV 6 同步点 | CrossCoreSetFlag/WaitFlag | 不迁移（Triton kernel launch） | 否（机制） |
| All-Cube/All-Vector 全核同步 | + PIPE_FIX/MTE3 | 不迁移 | 否（机制） |
| W^{DQ}/W^{UQ}/W^{DKV} NZ 格式 | FRACTAL_NZ layout | Triton layout（编译器） | 否（机制） |
| 拼接 matmul（W^{UQ}+W^{QR}） | 物理拼接 + split | `tl.concat` 后 dot，或两次 dot | 是（语义） |