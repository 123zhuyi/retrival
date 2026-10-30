# mla_preprocess - tiling 与 dispatch 分析

来源：`attention/mla_preprocess/`（commit `d28d8f9...`）。所有结论标路径+行号。tiling 参数数值属 `needs_verification`。

## A. 架构与 dtype 分发

`def.cpp` L118-119 注册 `ascend910b` + `ascend910_93`（A2/A3 训练/推理，README L7-8）。无 arch 子目录。

dtype 分发（`mla_preprocess.cpp` L17-22）：
- `ORIG_DTYPE_INPUT == DT_FLOAT16` -> `mla_preprocess_fp16.h`
- `ORIG_DTYPE_INPUT == DT_BF16` -> `mla_preprocess_bf16.h` + `mla_preprocess_no_quant.h`

3 个模板文件对应 dtype/quant 路径：`_fp16.h` / `_bf16.h` / `_no_quant.h`（浮点输出，no_quant）。

## B. tiling key 分发（cacheMode × 输出格式）

kernel 入口 `TILING_KEY_IS(...)` 分发（`mla_preprocess.cpp` L46+），模板参数 `<CacheMode, F1, F2, F3>`，F1/F2/F3 ∈ {ND, NZ}：

| tiling key 范围 | CacheMode | 含义 |
|---|---|---|
| 0,4,8,12,16,20,24,28 | `CACHE_MODE_KVCACHE` | kcache 与 q 拼接输出（cacheMode=0） |
| 32,36,40,... | `CACHE_MODE_KROPE_CTKV` | 拆分 kvCacheOut/krCacheOut + qOut/qRopeOut（cacheMode=1/3）；cacheMode=2 额外 NZ + int8 量化 |

tiling key = `cacheMode_offset + 4 × format_combo_index`（8 个 ND/NZ 组合，步长 4）。3 个 DataFormat 对应 q/kv/kr 三个输出的 ND/NZ 排布。`SetTilingKey(inDtype, param, doRmsQuant, context)`（tiling.h L57）在 host 侧设置。

## C. CV tiling 分离（Cube matmul + Vector norm/quant）

`MlaPreprocessTiling` 类（tiling.h L47-65）方法揭示 CV 分工：

| 方法 | 单元 | 操作 | 来源 |
|---|---|---|---|
| `RopeConcatTiling(param, aicNum)` | Cube (AIC) | 3 个 matmul + RoPE+Concat | tiling.h L54 |
| `RmsNormQuantTiling(numTokens, numVectorCore, hiddenState)` | Vector (AIV) | RmsNorm + Quant（2 次） | tiling.h L53 |
| `EinSumQuantTiling(param, aicNum, inDtype, doRmsQuant)` | Vector | EinSum 量化 | tiling.h L55 |

kernel 入口先 `ProcessCube()` 再 `ProcessVector()`（L52-53）。

## D. 核内 tiling（matmul + rmsnorm）

`MlaPpMatmulTilingData`（tilingdata.h L25-43）：m/k/n + m0/k0/n0（基本块）+ mLoop/kLoop/nLoop + coreLoop + swizzleCount/swizzleDirect + enShuffleK + blockDim + enLoadAllAmat + b0matPingPongBufferLen（ping-pong）。3 个 matmul 实例：mm1(W^DQKV)/mm2(W^UQ)/mm3(W^UK)（L51-53）。

`MlaTilingData`（L46-98）核心字段：
- `numCore`/`n`/`perTaskNum`/`resTaskNum`：核间切分（每核 perTaskNum，尾核 resTaskNum）。
- `mm1`/`mm2`/`mm3`：3 个 matmul tiling。
- `rms1`/`rms2`：2 次 RmsNorm（numCore/numCol/numRow/quantMin）。
- `esq*`（esqFrontCore/esqTailCore/esqHeadNum/esqColNum/...）：EinSumQuant 切分。
- `hiddenSizeQ`/`headNumQ`/`headDim`/`concatSize`/`rotaryCoeff`/`ntokens`。
- `realCore`/`nlCoreRun`/`lCoreRun`/`preCoreLoopTime`/`lastCoreLoopTime`：负载均衡（前核/尾核循环次数）。
- `maxWorkspaceSize`/`epsilon`/`doRmsNorm`/`qDownOutFlag`。

## E. workspace

kernel 入口 L41-43：`s1Gm = GetUserWorkspace(workspace)`，`s2Gm = s1Gm + maxWorkspaceSize`，`s3Gm = s2Gm + maxWorkspaceSize` -> 3 个等大 workspace section（中间 matmul/RmsNorm 结果）。`SetMlapoWorkSpace`（tiling.h L59）计算 `maxWorkspaceSize`。

## F. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| `numCore`/core id（tokenNum 切分） | `grid` + `tl.program_id` | token 轴映射 program |
| `perTaskNum`/`resTaskNum` | `mask = offs < n`（尾核） | 边界 |
| mm1/mm2/mm3 基本块 m0/k0/n0 | `tl.constexpr`（BLOCK_M/K/N） | 静态 meta，值 source-reported |
| 3 个 matmul（W^DQKV/W^UQ/W^UK） | 3 个 `tl.dot` | 数学迁移 |
| RmsNormQuant（γ/β/deqScale/quantOffset） | `tl.sum` + `rsqrt` + cast | 数学迁移 |
| RoPE（cos/sin） | 旋转复数乘 | 数学迁移 |
| Cache 写回（slot_mapping 索引） | `tl.store(cache + slot_mapping*...)` | scatter store |
| tiling key（cacheMode×format） | `if constexpr` / launcher dispatch | key 本身不迁移 |
| `*_iterator.h`（GM/L1/L0C/UB 搬运） | Triton load/store（编译器） | CANN 专有，不迁移 |
| ProcessCube/ProcessVector CV 分离 | 单 program family / 两阶段 | 抽象为 fused 计算 |
| swizzle / ping-pong / enShuffleK | 不迁移（CANN 调度） | 抽象为 tile 复用 |
