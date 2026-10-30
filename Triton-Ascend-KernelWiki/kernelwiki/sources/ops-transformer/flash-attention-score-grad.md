---
id: ops-transformer-flash-attention-score-grad
title: CANN ops-transformer FlashAttentionScoreGrad source snapshot
type: source
repo: cann/ops-transformer
path: attention/flash_attention_score_grad/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer FlashAttentionScoreGrad 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/flash_attention_score_grad/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（2026-07-09）。本仓库为本地制品目录，`fixture: true` 表示来源摘录用于知识库分析，不做整仓复制。该算子为 `FlashAttentionScore` 的反向（训练梯度），前向来源见 [[ops-transformer-flash-attention-score]]。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `attention/flash_attention_score_grad/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 远端: `https://gitcode.com/cann/ops-transformer.git`
- 抓取日期: 2026-07-29
- 许可证: CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

| 源文件（相对仓库根） | SHA-256 | 作用 |
|------|------|------|
| `attention/flash_attention_score_grad/README.md` | `98accff2ec6ac3bc5511754d6a07faae4e18c2b7eb65bca139a29b88886adc71` | 算子总览、反向公式、I/O 参数、约束 |
| `attention/flash_attention_score_grad/docs/FAG算子设计介绍.md` | `862ac3ebc27300fd6d5bae3c8ae3115a24d418a06e65b21868261f286186472b` | 6 步反向流程、3 阶段入口、arch32/arch35 tiling、流水、多模板、编程视角 |
| `attention/flash_attention_score_grad/op_host/flash_attention_score_grad_def.cpp` | `0e4cf00891fedda22ab2c678aa16a0db34576539cd79cc52a4a05bd2af9eb3d1` | 算子定义（输入输出 dtype/layout、属性、AICore 配置 ascend950/910b/910_93） |
| `attention/flash_attention_score_grad/op_host/flash_attention_score_grad_tiling.cpp` | `2f329873400bcba4dcf77ebef45d54fa605a79632ac0a12f9f433999919ade24` | tiling 入口、架构分发、空 tensor、TilingPrepare 平台参数 |
| `attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad.cpp` | `c841eb6e816822592cb31fd46f22c6d97af0469f509f396347eb06398db800b0` | kernel 入口、20 模板参数、模板 dispatch 树（if constexpr） |
| `attention/flash_attention_score_grad/op_kernel/arch32/flash_attention_score_grad_template_tiling_key.h` | `234ff0e762f601e52a2054c95bfa1097af2cacfc55a1c8aac0c7e69da2c023c1` | arch32 tiling key 编码、对齐特化枚举 |
| `attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_template_tiling_key.h` | `a91faa17041e640dcee72619fc2ef1e5b570a5cd78fd51f24f9080486d7d2d3a` | arch35 tiling key 编码（12 字段） |
| `attention/flash_attention_score_grad/op_host/arch32/flash_attention_score_grad_tiling_common.h` | `48ce7cf7cd6d09263a4518060da6d0fdf130cd69c705b6d7897e214c2beabe87` | arch32 枚举（InputIndex/AttrIndex/SparseMode/PseType）、PreParams/PostParams TilingData |
| `attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_common_regbase.h` | `b5561462e270bbbd168ab1bde97e292714c50aac2ac187862baf405f79661f34` | arch35 枚举（含 ds_scale/p_scale 索引） |
| `attention/flash_attention_score_grad/op_host/arch35/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp` | `eb18fe394be4ad9ed5792ac6f1ec4f422cd90d3da2475df8917b0546e69efdf7` | arch35 统一 tiling、基础块常量（BN2_MAX_S=128 等） |
| `attention/flash_attention_score_grad/op_kernel/arch35/flash_attention_score_grad_tiling_data_regbase.h` | `c7383b5fe26ef3ed5ea9a6400c117902931803edd5d144d8d51a6c5dfe4cad05` | arch35 TilingData 结构体 |
| `attention/flash_attention_score_grad/op_api/aclnn_flash_attention_score_grad.cpp` | `e478234fe401c2c26f2b654b6603eb8480a0bf533f6ddcda00cceb97b7b05c57` | aclnn 两段式接口 V1/V2/V3/V4/Unpadding V1-V5 |

## 3. 关键代码区域摘要

### 3.1 反向 6 步公式（来源: FAG算子设计介绍.md L9-14；README.md L48-60）

设 `S=Mask(QK^T/√d + pse, atten_mask)`，`P=Dropout(Softmax(S),keep_prob)`，`Y=PV`：

1. 重算 `p = SimpleSoftmax(Mask(Matmul(query, key^T) + pse_shift) * scale)`（结果存 UB）
2. `dp = Dropout(Matmul(dy, value^T))`
3. `ds = p * Sub(dp, FlashSoftmaxGrad(dy, attention_in))`（FlashSoftmaxGrad 入参 dy 与前向输出 attention_in=Y）
4. `dq = Matmul(ds, key) * scale`
5. `dk = Matmul(ds^T, query) * scale`
6. `dv = Matmul(Dropout(p)^T, dy)`

### 3.2 三阶段 Pipeline 入口（来源: FAG算子设计介绍.md L20-23）

以 SameAB 模板为例：`Process`（任务块分配）-> Cube1 `ComputeMM1`（步 1，重算 p）-> Vector1 `ComputeVec`（步 2-3，dp/ds）-> Cube2 `ComputeMMDqkv`（步 4-6，dq/dk/dv）。共 4 次 matmul：mm1=Q·K^T（重算）、mm2=dy·V^T、mm3=ds·K / ds^T·Q、mm4=P^T·dy。

### 3.3 架构分发（来源: flash_attention_score_grad_tiling.cpp L372-429）

`TilingFlashAttentionGradScore` 检测 `NpuArch::DAV_3510`（arch35/950PR/950DT）走 `RunEmptyTilingRegbase`（空）或 regbase tiling；其余架构走 `RunEmptyTiling`（空）或 `TilingRegistryArch::DoTilingImpl` 分发到 arch32 tiling 类。空 tensor tiling key `FAG_EMPTY_TILING_KEY=0`（arch32）/ regbase key（arch35），workspace 预留 100MB（`WORKSPACE_SIZE=100*1024*1024`，L51）。`TilingPrepare` 读取 aivNum/aicNum/npuArch/ubSize/l1Size/l0a/l0b/l0c/l2CacheSize 写入 `FlashAttentionScoreGradCompileInfo`（L411-420）。`KERNEL_TYPE_MIX_AIC_1_2`（1 Cube:2 Vector，kernel L501）。

### 3.4 kernel 模板 dispatch（来源: flash_attention_score_grad.cpp L488-901）

20 个模板参数：`UB0,UB1,Block,IsSameAB,DataType,Layout,Sparse,MatmulCfg,Mm12IsNZOut,Mm345IsNZOut,HasDropOut,HasPse,HasAttenMask,EnableL1Reuse,TNDS1Pingpong,S1TemplateType,S2TemplateType,DTemplateType,IsDeterministic,HasRope`。按 `UB0/UB1/Block/IsSameAB/IsDeterministic` 分发：空 tensor(0,0,0) / SameAB(4,3,4,IsSameAB=1, key 0x1434) / S1S2(4,3,4,0) / BN2(4,3,1, key 0x134) / NGS1S2_BN(9,9,1, key 0x199) / B(9,9,0, key 0x099) / Basic_det(9,9,9,Det=1, key 0x1999) / Basic(9,9,9, key 0x999)。按 dtype（FP16/BF16/FLOAT）特化，FP32 仅支持 S1S2/BN2/BN/B。Layout 编码 0/1/2/3（BSNGD 族/SBNGD/BNGS1S2/TND）。NZ 输出、L1 复用、RoPE、对齐特化（Aligned16/32/48/64/128/192、NotAligned30/60/72/88）由模板参数隔离。

### 3.5 arch32 vs arch35 tiling（来源: FAG算子设计介绍.md L29-72；regbase cpp L93-106）

- arch32（A2/910b/910_93）：AIC/AIV 分离，CV 通路 L2/GM；C-Tiling 128×1024、V-Tiling 8×1024、nRatio=8、1:16 数据配比；多模板（8 类：S1S2/SameAB/BN2/NGS1S2_BN/B/Basic/Basic_det/Empty）。
- arch35（950PR/950DT）：AIC/AIV 融合，CV 高速通路 L0C->UB、UB->L1；C 128×128、V 64×128（C:V=1:2 核数比）；统一模板（2 文件：非确定性 `kernel.h`/`kernel_base.h`、确定性 `kernel_deter.h`，BN2/BN2S2/BN2GS1S2 由模板参数隔离）。基础块常量 `BN2_MAX_S=128`、`BN2S2_MAX_S=1024`、`REGBASE_POST_BASE=128*128`、`RESERVED_WORKSPACE_SIZE=64KB`。

### 3.6 算子定义的 dtype/layout/属性（来源: flash_attention_score_grad_def.cpp）

- 输入：query/key/value（FP16/BF16/FLOAT/HIFLOAT8/FLOAT8_E5M2/FLOAT8_E4M3FN）、dy（FP16/BF16/FLOAT，FP8 路径下同 qkv）；可选 pse_shift、drop_mask、padding_mask、atten_mask(BOOL/UINT8)、softmax_max/softmax_sum(FLOAT)、softmax_in、attention_in、prefix、actual_seq_qlen/kvlen、q/kv_start_idx、d_scale_q/k/v/dy/o（FP8 反量化）、query_rope/key_rope、sink、ds_scale、p_scale。
- 输出：dq/dk/dv/dpse（FP16/BF16/FLOAT）、dq_rope/dk_rope、dsink(FLOAT)。
- 属性：`scale_value`(1.0)、`keep_prob`(1.0)、`pre_tockens`(INT_MAX)、`next_tockens`(INT_MAX)、`head_num`(必需)、`input_layout`(必需)、`inner_precise`(0)、`sparse_mode`(0)、`pse_type`(1)、`seed`(2)、`offset`(0)、`out_dtype`(0)、`softmax_in_layout`("")。
- AICore 配置：ascend950（opFile=flash_attention_score_grad_apt）、ascend910b、ascend910_93（opFile=flash_attention_score_grad）。FP8 仅 ascend950，且不可配 pse/dropMask/attenMask/keepProb（README L179-181）。

### 3.7 aclnn 接口版本（来源: docs/aclnnFlashAttentionScoreGrad*.md；op_api L1461-2352）

| 接口 | 架构支持 | 新增 |
|------|---------|------|
| V1 `aclnnFlashAttentionScoreGrad` | A2 训练 √ / 950PR × | 基线（pseType=1） |
| V2 | A2 √ / 950PR × | pseType 参数 |
| V3 | A2 √ / 950PR ×（950PR 暂不支持 sink） | sinkInOptional 输入、dsinkOut 输出 |
| V4 | 950PR √ / A2 × | 合并 V2+UnpaddingV2、varlen（多等/不等长）、Dropout 调整（自动生成 mask） |
| Unpadding V1-V5 | 变长/TND | actual_seq_qlen/kvlen、q/kv_start_idx；V5 含完整 RoPE |

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用:
- `wiki/kernels/flash-attention-score-grad.md`
- `wiki/techniques/flash-attention-backward-recompute.md`
- `artifacts/flash-attention-score-grad/`

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 和关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` 并按上述路径读取原文件。tiling 参数具体数值、UB/L1 占用、FP8/BF16 精度边界、aclnn 运行时契约等需结合硬件环境进一步验证，相关结论在引用 wiki 页中标记 `confidence: inferred`，未决项标记 `needs_verification`。本页为 `static_ingest` 静态分析，未运行/未验证/未 benchmark 算子。
