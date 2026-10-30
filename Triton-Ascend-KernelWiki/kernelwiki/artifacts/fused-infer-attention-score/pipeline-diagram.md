# fused_infer_attention_score - 流水编排分析

来源：`op_kernel/attn_infra/`（52 个 `.hpp`）+ `op_kernel/flash_attention_regular.h`/`_regular_decode.h` + `op_host/tiling*.cpp` + aclnn V1-V5 doc。FIA 的核心架构是 `attn_infra/` 分层模板库，MTE/BufferID/queue/event/ND↔NZ/Cube 物理搬运属 CANN 专有，仅抽象为 tile 级语义。

## 1. attn_infra 分层架构（FIA 核心，区别于 PFA/IFA）

FIA 的 op_kernel 不再用 IFA 那样的扁平 arch32/arch35 模板，而是用 header-only C++ 模板库 `attn_infra/` 分层抽象，对应 attention 的 QK->softmax->PV->rescale 流程：

| 分层 | 目录 | 职责 | 对应 attention 阶段 |
|---|---|---|---|
| **gemm** | `attn_infra/gemm/block/` | matmul：`block_mmad_qk`=Bmm1(Q·K^T)，`block_mmad_pv`=Bmm2(P·V)；各有 `_decode` 变体 | Cube1 / Cube2 |
| **gemm tile_common** | `attn_infra/gemm/tile_common/` | Cube 数据搬运：`copy_gm_to_l1`、`copy_l1_to_l0a/l0b`、`copy_l0c_to_gm`、`tile_mmad` | Cube 输入/输出 |
| **epilogue** | `attn_infra/epilogue/block/` | softmax/rescale/writeback：`block_epilogue_online_softmax`=FlashSoftmax，`block_epilogue_rescale_o`=÷softmax_sum+写回；`CombineScale`=量化 scale 合并；各有 `_low_prec` 变体 | Vector1 / Vector2 |
| **epilogue tile_common** | `attn_infra/epilogue/tile_common/` | Vector 搬运+elemwise：`copy_gm_to_ub`、`copy_ub_to_gm`、`tile_broadcast_*`、`tile_cast`、`tile_elemwise_*`、`tile_swizzle` | pse/scale/mask/rescale |
| **layout** | `attn_infra/layout/` | ND/NZ 布局：`matrix`、`vector` | 数据排布（CANN 专有，不迁移） |
| **arch** | `attn_infra/arch/` | 硬件抽象：`resource`（UB/L1/L0 资源）、`local_tensor_buffer`、`cross_core_sync` | buffer/sync（CANN 专有，不迁移） |
| **common** | `attn_infra/`（coord/gemm_coord/matrix_coord/base_defs）+ `detail/` | 坐标/基础定义/工具 | 通用 |

> Triton 抽象：`gemm` -> `tl.dot`；`epilogue online_softmax` -> running max/sum + `exp(m_old-m_new)` rescale；`epilogue rescale_o` -> `o=acc/l_i`；`layout`/`arch` -> Triton 编译器管理，不迁移。

## 2. 四阶段流程（attn_infra gemm + epilogue 交替）

| 阶段 | attn_infra 模块 | 操作 | 数据依赖 |
|---|---|---|---|
| Cube1 (Bmm1) | `gemm/block/block_mmad_qk` | Q·K^T -> attention_score（FP32 累加） | Q, K |
| Vector1 (Vec1) | `epilogue/tile_common`（elemwise）+ `epilogue/block/block_epilogue_online_softmax` | pse add/mul -> ×scale -> atten_mask select -> FlashSoftmax（在线 max/sum） | Bmm1 结果, pse, mask |
| Cube2 (Bmm2) | `gemm/block/block_mmad_pv` | P·V | Vec1 结果, V |
| Vector2 (Vec2) | `epilogue/block/block_epilogue_rescale_o` | rescale(÷softmax_sum) -> 可选 quant -> 写回 attention_out；可选 softmaxLse 输出 | Bmm2 结果 |

依赖链：`Bmm1 -> Vec1 -> Bmm2 -> Vec2`。decode 场景用 `_decode` 变体（`block_mmad_qk_decode`/`block_mmad_pv_decode`），配合 FlashDecode。

## 3. prefill vs decode 模板差异

| 模板 | 文件 | 场景 | S1 | FlashDecode |
|---|---|---|---|---|
| regular | `flash_attention_regular.h`（1059 行） | prefill 全量 | S1>1（多 token） | 否 |
| regular_decode | `flash_attention_regular_decode.h`（745 行） | decode 增量 | S1=1 | 是（split-KV + lse 规约） |

两模板共享 attn_infra gemm/epilogue 骨架，差异在 S1 维度处理与 FlashDecode 规约。`fused_infer_attention_score.cpp`（arch32 入口）按 tiling key 分发；arch35 经 `_template_tiling_key.h`（3116 行）`if constexpr` 分发。

## 4. 在线 FlashSoftmax（`block_epilogue_online_softmax.hpp`，2050 行）

FlashSoftmax 对 Skv 轴切分，在线刷新：

```
for i in range(SkvSplits):
    s_i = QK[:, s2_start:s2_end] · scale
    s_i = mask_select(s_i, atten_mask)            # true -> -inf
    m_ij = row_max(s_i); s_i -= m_ij; p_i = exp(s_i); l_ij = row_sum(p_i)
    if i == 0:
        m_i = m_ij; l_i = l_ij; acc = p_i · V_i
    else:
        alpha = exp(m_i - m_ij)                    # rescale factor
        l_i = l_i * alpha + l_ij
        acc = acc * alpha + p_i · V_i
        m_i = max(m_i, m_ij)
attention_out = acc / l_i                          # rescale_o
softmaxLse = log(l_i) + m_i                        # 可选输出（softmaxLseFlag）
```

`block_epilogue_online_softmax_low_prec`（867 行）为低精度变体；`block_epilogue_rescale_o`（1102 行）负责 ÷softmax_sum + 写回；`CombineScale`（281 行）合并量化 scale。

## 5. FlashDecode（decode 场景，`regular_decode`）

decode（Q_S=1）时，BN2 不足 0.4×核数则对 S2 外切 SplitKv，各核独立 Attention，最后 lse 加权规约（同 IFA）：

```
scale[i] = exp(lse[i]) / Sum(exp(lse[i]))
res = Σ split_res[i] · scale[i]
```

可迁移为 Triton 两段式（split kernel 输出 partial lse + partial o；combine kernel 加权 reduce）。精确 lse 公式 `needs_verification`（以 `block_epilogue_online_softmax.hpp` 为准）。

## 6. 量化 / 反量化（V5 分离 KV 反量化）

| 模式 | 参数 | 说明 |
|---|---|---|
| 全量化 | deqScale1/quantScale1/deqScale2/quantScale2/quantOffset2 | BMM1 后反量化 / BMM2 前量化 / 输出量化 |
| 伪量化（V4-） | antiquantScale/antiquantOffset | KV 反量化（V5 不支持） |
| **分离 KV 反量化（V5）** | keyAntiquantScale/Offset、valueAntiquantScale/Offset | K/V 各自反量化，支持 per-tensor/channel/token/token+head/token-group + PA 模式 |
| query 量化（V5） | queryQuantMode、dequantScaleQuery | Q 量化模式 |

`CombineScale`（epilogue）合并量化 scale。FP8/FP4/INT4 精度边界 `needs_verification`。

## 7. sharedPrefix / RoPE / learnableSink（V5）

| 特性 | 参数 | Triton 抽象 |
|---|---|---|
| sharedPrefix | keySharedPrefix/valueSharedPrefix/actualSharedPrefixLen | 前缀 KV 独立 load + 与主 KV 拼接 |
| RoPE 解耦 | queryRope/keyRope（+ keyAntiquantRopeScale） | Q/K 沿 D 拼接 nope+rope 分段 dot |
| learnableSink | learnableSink | sink 项加入 attention 分母 |
| q/kv StartIdx | qStartIdx/kvStartIdx（V5 新增） | Q/KV 序列起始偏移 |
| query/kv padding | queryPaddingSize/kvPaddingSize | 起始偏移 = `Smax - padding - actualLen`，越界 0 |

## 8. UB / L1 / L0 内存（attn_infra arch + gemm copy）

| 区域 | attn_infra 模块 | 用途 |
|---|---|---|
| L1 | `gemm/tile_common/copy_gm_to_l1` | Cube 输入缓冲（Q/K/V 块） |
| L0A/L0B | `copy_l1_to_l0a`/`copy_l1_to_l0b` | Cube matmul 输入 |
| L0C | `copy_l0c_to_gm` | Cube matmul 输出（FP32 累加） |
| UB | `epilogue/tile_common/copy_gm_to_ub`/`copy_ub_to_gm` | Vector 输入/输出（pse/mask/softmax 中间） |
| resource | `arch/resource.hpp` | UB/L1/L0 资源管理 |
| sync | `arch/cross_core_sync.hpp` | 核间同步（FlashDecode 规约） |

精确字节预算 `needs_verification`。

## 9. Workspace

`workspace = sysWorkspace + coreNum × (softmax lse/max/sum + mm1Res + mm2Res)` + PA/antiquant/sharedPrefix 扩展（`tiling.cpp` L1536-1537）。FlashDecode 额外需 split 数 × partial lse/o。`needs_verification`。

## 10. FIA 相对 PFA / IFA 的差异

| 差异 | PFA（prefill） | IFA（decode） | FIA（统一） |
|---|---|---|---|
| 场景 | 推理 prefill | 推理增量 decode | **prefill + decode 统一** |
| query S1 | 1~1M | 固定 1 | Q_S>1 或 =1（双场景） |
| 模板组织 | arch32 多模板 + arch35 服务化 | arch32 三模板 + arch35 服务化 | **attn_infra 分层 + regular/regular_decode** |
| FlashDecode | 无 | 有 | 有（decode 分支） |
| softmaxLse 输出 | 有（供 decode） | 无 | 有（softmaxLseFlag） |
| sharedPrefix | V3+ | 无 | **V5** |
| RoPE 解耦 | 部分 | 无 | **V5（queryRope/keyRope）** |
| 分离 KV 反量化 | 无 | 无 | **V5（key/value AntiquantScale）** |
| learnableSink | 无 | 无 | **V5** |
| FP8/FP4 | V3+ | 无 | **V5（HIFLOAT8/FLOAT8_E4M3FN/FLOAT4_E2M1）** |
| 版本 | V1-V3 | V1-V4 | **V1-V5** |
| 310P | × | √ | × |

## 11. 阶段 -> Triton 抽象

| 阶段 | CANN 机制（attn_infra） | Triton 抽象 | 可迁移 |
|---|---|---|---|
| Bmm1 QK | `gemm/block_mmad_qk` + L0C FP32 累加 | `tl.dot(q, k.T)` FP32 | 是（数学） |
| pse/scale/mask | `epilogue/tile_common` elemwise | `s = s+pse; s*=scale; tl.where(mask,-inf,s)` | 是 |
| FlashSoftmax | `epilogue/block_epilogue_online_softmax` | running max/sum + `exp(m_old-m_new)` rescale | 是（算法） |
| Bmm2 PV | `gemm/block_mmad_pv` | `tl.dot(p, v)` | 是（数学） |
| rescale 写回 | `epilogue/block_epilogue_rescale_o` | `o = acc / l_i` | 是 |
| FlashDecode | `regular_decode` split-KV + lse 规约 | 两段 kernel（split + combine） | 是（算法） |
| sharedPrefix | 前缀 KV 拼接 | 独立 load + 拼接 | 是（语义） |
| RoPE | nope+rope 分段 dot | D 维拼接 constexpr + 分段 dot | 是（语义） |
| 分离 KV 反量化 | K/V 各自 cast+scale | per-token/head cast 融合 | 是（语义，精度 needs_verification） |
| layout ND/NZ | `layout/matrix` | Triton layout（编译器） | 否（机制） |
| arch buffer/sync | `arch/resource`/`cross_core_sync` | Triton 编译器管理 | 否（机制） |
| Cube L1/L0 搬运 | `gemm/tile_common/copy_*` | Triton load/store（编译器） | 否（机制） |
