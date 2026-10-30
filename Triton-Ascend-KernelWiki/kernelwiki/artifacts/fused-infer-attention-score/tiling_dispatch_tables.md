# fused_infer_attention_score - tiling 与 dispatch 分析

来源：`attention/fused_infer_attention_score/`（commit `d28d8f9…`）。FIA 无 `算子设计介绍.md`，tiling/pipeline 知识来自 `tiling.cpp`/`tiling_v3.cpp`/`tiling_v2.cpp` 源码与 aclnn V1-V5 doc。tiling 参数具体数值与精确行号属 `needs_verification`。FIA 统一 prefill（Q_S>1）+ decode（Q_S=1），架构分 arch32（A2/A3）与 arch35（950）。

## A. 架构分发（host -> kernel）

`def.cpp` L568-571 / L2048-2050 注册 AICore config：

| 架构 | SocVersion / config | opFile（kernel 入口） | tiling 路径 | 说明 |
|---|---|---|---|---|
| Atlas A2 训练/推理 | `ascend910b` L570 | `fused_infer_attention_score`（arch32） | `tiling.cpp` + `arch32/tiling_v3.cpp` | regular/regular_decode 模板 |
| Atlas A3 训练/推理 | `ascend910_93` L571 | `fused_infer_attention_score`（arch32） | arch32 | 同 A2 |
| Ascend 950PR/950DT | `ascend950` L2050 | `fused_infer_attention_score_apt`（arch35） | `arch35/tiling_v2.cpp` | `attn_infra/` 分层模板 |

> FIA 不支持 310P（README L11 推理加速卡 ×）。V5 接口仅 950（V5 doc L10-15），V4 及之前跨 A2/A3/950。

## B. arch32 tilingkey（`tilingkey.h` 259 行，十进制组装）

arch32 用十进制 tilingkey 宏（`fused_infer_attention_score_tilingkey.h`），组装维度与 IFA/FA 同源（layout / Q dtype / KV dtype / output dtype / splitKv(FlashDecode) / PA / antiquantMode / perfMode(模板) / mode）。**具体字段表 `needs_verification`**（FIA 无设计介绍文档，未逐一提取）。`GenTilingKey` 在 `tiling.cpp` L923-924 / L1548。

> 该十进制 tilingkey 是 CANN 编译期模板注册机制，**不迁移**；Triton 用 `if constexpr`/launcher dispatch。

## C. arch35 模板参数化（`template_tiling_key.h` 3116 行）

arch35 用结构化模板参数做 `if constexpr` 分发（`fused_infer_attention_score_template_tiling_key.h`，3116 行），参数维度：layout / S1 / S2 / D / ImplMode / PseMode / QuantMode / Mask / MatMulType / SplitCoreMode（与 PFA/IFA arch35 同源）。`arch35/tiling_v2.cpp` L1022-1025 / L1145-1148 生成模板 tiling key。**具体枚举值与 ConfigValue 组合 `needs_verification`**（未逐一提取）。

## D. 核间切分（core partition）

| 项 | 结论 | 来源 |
|---|---|---|
| 逻辑轴 | B、N2（kv head）、G（q/kv head 比）、S1（q seq）、S2（kv seq） | `tiling.cpp`（同 FA/IFA） |
| prefill（Q_S>1） | S1 参与切分（多 token） | README L93-154 |
| decode（Q_S=1） | S1=1 不切；BN2 不足时 S2 外切 SplitKv（FlashDecode） | README L158-165；`tiling.cpp` L1426-1439 |
| FlashDecode | split-KV 各核独立 Attention -> lse 加权规约 | `tiling.cpp` L1426-1439；`flash_attention_regular_decode.h` |
| 分核 | `tiling.cpp` L291-1548 | 静态读取 |
| arch 分发 | `tiling.cpp` L100-247 | 静态读取 |

## E. 核内 tiling 基本块

| 单元 | 推断 | 状态 |
|---|---|---|
| Cube Bmm1 (Q·K^T) | prefill: BLOCK_M×BLOCK_N；decode: 1×BLOCK_N；FP32 累加 | needs_verification（数值） |
| Cube Bmm2 (P·V) | BLOCK_N×DV，FP32 累加 | needs_verification |
| Vector V-Tiling | 沿 S2 分块，FP32 中间（~32KB） | needs_verification |
| GQA G 切分 | UB 驱动 | needs_verification |

## F. 模板选择 + 特性触发

| 分支 | 触发条件 | 来源 |
|---|---|---|
| regular 模板（prefill） | Q_S>1 全量场景 | `flash_attention_regular.h`（1059 行） |
| regular_decode 模板（decode） | Q_S=1 增量场景，含 FlashDecode | `flash_attention_regular_decode.h`（745 行） |
| FlashDecode（splitKv） | decode + BN2 不足 | `tiling.cpp` L1426-1439 |
| MLA | `tiling.cpp` L250-263 | `tiling.cpp` |
| PagedAttention | `block_table` 非空 | `tiling.cpp` L174-189；V5 doc L348-362 |
| sharedPrefix | `keySharedPrefix`/`valueSharedPrefix` 非空 | V5 doc L464-499 |
| 分离 KV 反量化（V5） | `keyAntiquantScale`/`valueAntiquantScale` 非空（替代 antiquantScale/Offset） | V5 doc L394-463 |
| RoPE 解耦 | `queryRope`/`keyRope` 非空 | V5 doc L80-81 |
| softmaxLse 输出 | `softmaxLseFlag=true` | V5 doc 属性 |
| 空 query/KV | query 空 -> 返回；KV 空 -> 输出全零 | README L89 |

## G. Workspace

`workspaceSize` 由 `tiling.cpp` L1536-1537 计算（sysWorkspace + coreNum × softmax/max/sum + mm1Res + mm2Res + PA/antiquant 扩展）。FlashDecode 场景额外需 split 数 × partial lse/o。**精确字节预算 `needs_verification`**。

## H. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| `blockNum`/core id 沿 BN2 | `grid` + `tl.program_id` | B/N2/G/S1 轴映射 |
| prefill S1>1 / decode S1=1 | `BLOCK_M` constexpr（Q_S 决定） | 统一 kernel 内分支 |
| FlashDecode split-Kv | `tl.program_id(split)` + combine kernel | 两段式 |
| S2 基本块 | `BLOCK_N: tl.constexpr` | 核内循环步长 |
| **attn_infra gemm（block_mmad_qk/pv）** | `tl.dot(q,k.T)` / `tl.dot(p,v)` | Bmm1/Bmm2 |
| **attn_infra epilogue（online_softmax/rescale_o）** | running softmax + `o=acc/l_i` | FlashSoftmax + 写回 |
| **attn_infra layout（ND/NZ）** | Triton layout（编译器管理） | 不迁移 |
| **attn_infra arch（buffer/sync）** | Triton 编译器管理 | 不迁移 |
| TilingKey 分支 | `if constexpr` / 独立 kernel / launcher | key 不迁移 |
| regular / regular_decode 模板 | 2 个 `@triton.jit` kernel（prefill/decode）或同 kernel 分支 | 按 Q_S 选 |
| sharedPrefix | 前缀 KV 独立 load + 拼接 | 间接 load |
| RoPE 解耦 | Q/K 沿 D 拼接 nope+rope 分段 dot | constexpr 拼接 |
| 分离 KV 反量化 | K/V 各自 cast+scale 融合 | per-token/head |
| PA blockTable | `tl.load(block_table)` 间接 load | block_size 对齐 |
| tail / 变长 actualSeqLen | `mask = offs < actual_len` | 边界 |
| workspace | 中间 `tl` tensor 或丢弃 | Triton 管理临时 |
| CV/MTE/BufferID/queue/event | 不迁移 | 抽象为 tile 复用/fused load-store |
