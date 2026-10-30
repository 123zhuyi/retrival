# mla_prolog_v3 - tiling 与 dispatch 分析

来源：`attention/mla_prolog_v3/`（commit `d28d8f9…`）。设计文档（454 行）完整。MLA Prolog V3 为 DeepSeek V2/V3 核心 MLA 的前处理融合算子（4 路 matmul + 2 RmsNorm + 2 RoPE + Cache 更新）。三架构（arch32 A2/A3 + arch35 950）。

## A. 架构分发

`def.cpp` L171-172 / L315 注册 AICore config：

| 架构 | SocVersion | config | kernel 入口 | 说明 |
|---|---|---|---|---|
| Atlas A2 训练/推理 | `ascend910b` L171 | `mla_prolog_v3` | `mla_prolog_v3.cpp`（arch32） | 共享 `mla_prolog/` kernel |
| Atlas A3 训练/推理 | `ascend910_93` L172 | 同上 | 同上 | 同 A2 |
| Ascend 950PR/950DT | `ascend950` L315 | `mla_prolog_v3`（arch35 单独 config） | `mla_prolog_v3.cpp`（arch35 模板） | 950 单独 dtype/layout（FLOAT8_E4M3FN） |

## B. TilingKey（26 位二进制，9 字段）

来源：设计介绍 L70-85。`mla_prolog_v3` kernel 模板直接对应为 9 个 uint8_t 模板参数（`mla_prolog_v3.cpp` L25-27）。

| 二进制位 | 模板参数 | 说明 | 取值 |
|---|---|---|---|
| 0-3 | CACHE_MODE | KVCache 存储格式 | 0:BNSD(预留) / 1:PA_BSND / 2:PA_NZ |
| 4-5 | SCENARIO | 输入场景 | 0:FP16(预留) / 1:非量化 / 2:量化 |
| 6-9 | QUANT_MODE | 量化模式 | 0:非量化 / 1:MMQcQr量化 / 2:+KVcache量化 / 3:MMCqCkvkr+MMQcQr / 4:全量+KVcache / 5:MMQcQr+KVcache pertile / 6:全量+KVcache pertile / 7:Mxfp8+MMQcQr / 8:Mxfp8+MMQcQr+KVcache |
| 10 | ENABLE_DEQUANT_OPTIONAL | 反量化使能 | 0:关闭 / 1:开启 |
| 11 | ENABLE_GROUP_COMPUTE_OPTIONAL | 量化算力分组优化 | 0:关闭 / 1:开启 |
| 12-13 | EMPTY_TENSOR_MODE | 空 tensor 场景 | 0:无空 / 1:KVCache+KRCache空 / 2:Query空 |
| 14-15 | ACTUAL_SEQ_LEN_MODE | actualSeqLen 使能 | 0:关闭 / 1:使能 |
| 16-17 | SPLIT_M_MODE | 切 M 模式 | 0:关闭(切N) / 1:使能(切M) |
| 18-25 | CV_MODE | CV 模式 | 6:ASCENDC_TPL_MIX_AIC_1_1 / 7:ASCENDC_TPL_MIX_AIC_1_2 |

## C. 核间切分（4 路 matmul 各不同）

| matmul | 切分轴 | 切分方式 | 来源 |
|---|---|---|---|
| MatmulCq (x·W^{DQ}) | Hcq（headSizeCq） | 按 AIC 核数切分 N 轴 | 设计介绍 L39-40 |
| MatmulCkvKr (x·[W^{DKV}\|W^{KR}]) | Hckv（headSizeCkvKr） | **固定分成 64 块** | 设计介绍 L42 |
| MatmulQcQr (c^Q·[W^{UQ}\|W^{QR}]) | Hqc（headSizeQcQr） | 按 AIC 核数切分 N 轴 | 设计介绍 L44 |
| MatmulQn (q^C·W^{UK}) | 不切 N | 单点 | 设计介绍 L45 |

分核原则（设计介绍 L29-36）：
a. 核数用满，防止部分核闲置。
b. 各核负载均匀，避免计算量不均。
c. AIC 和 AIV 间数据量匹配算力，避免长时间空闲。

## D. 核内处理

| 阶段 | 说明 |
|---|---|
| 1. Cube MatmulCq | x · W^{DQ} -> c^Q [BS, Hcq] |
| 2. Vector RmsNormCq | RmsNorm(c^Q) -> c_norm^Q [BS, Hcq] |
| 3. Cube MatmulCkvKr | x · [W^{DKV}\|W^{KR}] -> [c^{KV}\|k^R] [BS, Hckv+Dr] |
| 4. Vector RmsNormCkv + RoPE k^R | RmsNorm(c^{KV}) + RoPE(k^R) -> c_norm^{KV} + k^R_rope |
| 5. Cube MatmulQcQr | c_norm^Q · [W^{UQ}\|W^{QR}] -> [q^C\|q^R] [BS, Hckv+Dr] |
| 6. Vector RoPE q^R | RoPE(q^R) -> q^R_rope |
| 7. Cube MatmulQn | q^C · W^{UK} -> q^N [BS, N, Hckv] |
| 8. Vector (KVCache update) | 更新 kv_cache + kr_cache |

## E. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| MatmulCq (x·W^{DQ}) | `tl.dot(x, W_dq)` | 下采样 |
| MatmulCkvKr (x·[W^{DKV}\|W^{KR}]) | `tl.dot(x, W_dkv_kr)` -> split | 拼接后 split |
| MatmulQcQr (c^Q·[W^{UQ}\|W^{QR}]) | `tl.dot(c_q, W_uq_qr)` -> split | 拼接后 split |
| MatmulQn (q^C·W^{UK}) | `tl.dot(q_c, W_uk)` | 上采样 |
| RmsNorm | `x * rsqrt(mean(x^2) + eps) * gamma` | 可迁移 |
| RoPE | 偶奇拆分 * cos + 翻转 * sin | 可迁移 |
| KVCache scatter | `tl.store(cache + cache_index, ...)` | 间接 store |
| TilingKey 9 模板参数 | `if constexpr` / 独立 kernel | 不迁移 key 本身 |
| CV 6 同步点 | 不迁移（Triton kernel launch） | CANN 专有 |
| W^{DQ}/W^{UQ}/W^{DKV} NZ 格式 | Triton layout（编译器） | CANN 专有 |