# mla_prolog - tiling 与 dispatch 分析

来源：`attention/mla_prolog/`（commit `d28d8f9...`）。所有结论标路径+行号。

## A. 架构分发

`def.cpp` L145-146：`ascend910b` + `ascend910_93`（A2/A3，BF16/INT8）。L148-267：`ascend950` 独立 config（950，额外 FP8_E4M3FN 支持，7 组 dtype 组合）。README L7-9：950 不支持，A3+A2 支持。

## B. tiling key 二进制编码

`mla_prolog_template_tiling_key.h` L68-79（设计介绍 L70-79）：

| 位域 | 变量名 | 宽度(bits) | 说明 |
|---|---|---|---|
| 0-3 | CACHE_MODE | 4 | 0=BNSD(预留), 1=PA_BSND, 2=PA_NZ |
| 4-5 | SCENARIO | 2 | 0=FP16(预留), 1=NO_QUANT, 2=QUANT |
| 6-9 | QUANT_MODE | 4 | 0=MMQcQr量化, 1=MMQcQr量化+KVCache量化 |
| 10 | ENABLE_DEQUANT_OPTIONAL | 1 | 反量化使能（与 GROUP_COMPUTE 互斥） |
| 11 | ENABLE_GROUP_COMPUTE_OPTIONAL | 1 | 分组算力优化 |
| 12-13 | EMPTY_TENSOR_MODE | 2 | 0=无空tensor, 1=KV空, 2=Query空 |

kernel 入口模板参数 8 个（`mla_prolog.cpp` L20-22）：`CacheMode, Scenario, QuantMode, EnableDequantOpt, EnableGroupComputeOpt, EmptyTensorMode, ActualSeqLenMode, SplitMMode, CvMode`。

## C. 场景分支（3 量化场景）

kernel 入口（`mla_prolog.cpp` L69-99）：

| SCENARIO | QUANT_MODE | 含义 | dtype 组合 |
|---|---|---|---|
| NO_QUANT(1) | - | 非量化 | BF16×BF16×BF16 |
| QUANT(2) | PARTIAL_QUANT_KV_NO_QUANT(0) | weight_uq_qr 量化, KV 非量化 | BF16×INT8×BF16 |
| QUANT(2) | PARTIAL_QUANT_KV_QUANT_PER_CHANNEL(1) | weight_uq_qr 量化 + KV per-channel 量化 | BF16×INT8×INT8 |

## D. 4 matmul 切分（设计介绍 L36-44）

| matmul | 公式 | M 轴 | N 轴 | K 轴 | 核间切分 |
|---|---|---|---|---|---|
| MatmulCq | x·W^DQ | BS 合轴 | Hcq | He | 按 Hcq 按 Cube 核数 |
| MatmulCkvKr | x·[W^DKV\|W^KR] | BS 合轴 | Hckv+Dr | He | 按 Hckv+Dr 分 64 块 |
| MatmulQcQr | cQ_norm·[W^UQ\|W^QR] | BS 合轴 | N*(D+Dr) | Hcq | 按 N*(D+Dr) 按 Cube 核数 |
| MatmulQn | qC·W^UK | BS 合轴 | Hckv | Hcq | 按 Hckv |

## E. CV 流水同步（6 sync event，设计介绍 L47-67）

| 阶段 | 同步 event | 说明 |
|---|---|---|
| MatmulCq → RmsNormCq | SYNC_MMCQ_NOMRCQ | Cube 写 Cq 结果、Vector 读 |
| MatmulCkvKr → RmsNormCkv | SYNC_MMCKVKR_NOMRCKV | Cube 写 CkvKr 结果、Vector 读 |
| RmsNormCq → MatmulQcQr | SYNC_NOMRCQ_MMQCQR | Vector 写 normCq、Cube 读 |
| MatmulQcQr → RoPEQr | SYNC_MMQCQR_ROPEQR | Cube 写 QcQr、Vector 读 |
| MatmulQcQr 全核同步 → MatmulQn | SYNC_ALL_CUBE | 所有 Cube 核完成后启动 MatmulQn（分核不同） |
| Vector 全核同步 | SYNC_ALL_VECTOR | 确保数据搬运 |

## F. 核内/核间切分

`MlaPrologTilingData`/`MlaPrologBaseParams`（`tiling_data.h`）：BS 合轴做 M 轴，token 分配 `numCore`/`perTaskNum`/`resTaskNum`。`MIX_AIC_1_2`（kernel 入口 L51）按 1:2 CV 配比。`CvMode` 模板参数控制 `cvRatio=1 or 2`（L60）。

## G. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| BS 合轴 M 维 | `grid` + `tl.program_id` | 沿 token 映射 |
| 4 matmul 基本块 | `tl.constexpr`（BLOCK_M/K/N） | 静态 meta |
| tiling key 二进制位 | `if constexpr` / launcher 分支 | 位域编码不迁移 |
| 3 场景 dtype | 独立 kernel / `if constexpr` | 分支逻辑迁移 |
| RmsNorm + RoPE + Cache scatter | `tl.sum`+`rsqrt`+旋转乘+scatter store | 数学迁移 |
| 6 CV sync event | 不迁移（CANN 手动同步） | 抽象为 tile 依赖 |
| 950 FP8 支持 | 配置型 dtype 分支 | 条件迁移 |
| 空 tensor 分支 | early-return | 边界迁移 |